"""Notices crashes (crash_detect.py) and remembers where they happened.

  in:  /cmd_vel_safe, and while driving /imu/data, /wheel/odom, TF
       /crash/remove       geometry_msgs/Point: the mark at x, y (NaN = all of them)
  out: /crash/event        std_msgs/String JSON, on each crash; drive_guard stops for it
       /crash/marks        sensor_msgs/PointCloud2 in map, for the costmaps' crash_layer
       /crash/list         latched JSON: the marks on the current map

A crash usually means something the lidar can't see, so marks live in their own costmap layer that the
lidar doesn't clear. They're saved per map (maps/<map>.crash.json; while mapping maps/.pending_crash.json,
which nav_manager renames when the map is saved) and kept out of the map image: a false alarm or a
moved object can just be removed. While mapping, SLAM is paused for a moment so scans from the knock
don't smear the map.
"""
import json
import math
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Twist
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, PointCloud2, PointField
from slam_toolbox.srv import Pause
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from robot_bringup.crash_detect import CrashDetector, load_marks, mark_at, mark_points, remove_marks, save_marks

WAKE_S = 5.0        # keep listening this long after the last motion command
SLAM_PAUSE_S = 2.0  # long enough for the robot to stop rocking


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class CrashMonitor(Node):
    def __init__(self):
        super().__init__('crash_monitor')
        self.map_dir = os.path.expanduser(self.declare_parameter('map_dir', '~/robot_ws/maps').value)
        self.det = CrashDetector()
        self.tf, self.tf_listener, self.subs, self.imu_yaw = None, None, None, None
        self.moved_at, self.mode, self.nav_map = 0.0, 'unknown', ''
        self.marks, self.marks_path = [], None
        self.slam_resume_at = None
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)
        self.event_pub = self.create_publisher(String, '/crash/event', 10)
        self.list_pub = self.create_publisher(String, '/crash/list', latched)
        self.cloud_pub = self.create_publisher(PointCloud2, '/crash/marks', 1)
        self.create_subscription(Twist, '/cmd_vel_safe', self.on_cmd, 10)
        self.create_subscription(Point, '/crash/remove', self.on_remove, 10)
        self.create_subscription(String, '/robot/nav_mode', self.on_mode, latched)
        self.create_subscription(String, '/robot/nav_map', self.on_map, latched)
        self.slam_pause = self.create_client(Pause, '/slam_toolbox/pause_new_measurements')
        self.clear_clients = [self.create_client(ClearEntireCostmap, f'/{c}_costmap/clear_entirely_{c}_costmap')
                              for c in ('global', 'local')]
        self.publish_list()
        self.create_timer(1.0, self.tick)
        self.create_timer(0.1, self.resume_slam)

    # ---------- which map ----------
    def current_path(self):
        if self.mode in ('mapping', 'exploring'):
            return os.path.join(self.map_dir, '.pending_crash.json')
        if self.mode == 'navigating' and self.nav_map:
            return os.path.join(self.map_dir, f'{self.nav_map}.crash.json')
        return None

    def on_mode(self, msg):
        self.mode = msg.data
        self.slam_resume_at = None  # SLAM was restarted or stopped
        self.reload()

    def on_map(self, msg):
        self.nav_map = msg.data
        self.reload()

    def reload(self):
        path = self.current_path()
        if path != self.marks_path:
            self.marks_path = path
            self.marks = load_marks(path) if path else []
            self.publish_list()

    def publish_list(self):
        self.list_pub.publish(String(data=json.dumps(self.marks)))

    # ---------- sensors while driving ----------
    def on_cmd(self, msg):
        now = time.monotonic()
        self.det.on_cmd(now, msg.linear.x)
        if abs(msg.linear.x) > 1e-3 or abs(msg.angular.z) > 1e-3:
            self.moved_at = now
            self.wake()

    def wake(self):
        if self.subs is not None:
            return
        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)
        self.subs = [self.create_subscription(Imu, '/imu/data', self.on_imu, qos_profile_sensor_data),
                     self.create_subscription(Odometry, '/wheel/odom', lambda m: self.det.on_wheel(m.twist.twist.linear.x), 10)]

    def sleep(self):
        for sub in self.subs:
            self.destroy_subscription(sub)
        self.tf_listener.unregister()
        self.subs, self.tf, self.tf_listener, self.imu_yaw = None, None, None, None

    def on_imu(self, msg):
        if self.imu_yaw is None:  # how the board is mounted
            try:
                self.imu_yaw = yaw_of(self.tf.lookup_transform('base_link', msg.header.frame_id, Time()).transform.rotation)
            except Exception:  # noqa: BLE001 - TF not there yet; next message
                return
        a = msg.linear_acceleration
        forward = math.cos(self.imu_yaw) * a.x - math.sin(self.imu_yaw) * a.y
        crash = self.det.on_accel(time.monotonic(), forward)
        if crash:
            self.on_crash(*crash)

    # ---------- a crash ----------
    def on_crash(self, direction, jolt):
        self.get_logger().warning(f'crash detected ({"forward" if direction > 0 else "backward"}, jolt {jolt:.1f} m/s^2)')
        marked = self.add_mark(direction, jolt)
        self.event_pub.publish(String(data=json.dumps({'direction': direction, 'jolt': round(jolt, 1), 'marked': marked})))
        self.pause_slam()

    def add_mark(self, direction, jolt):
        if self.marks_path is None:
            return False
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time()).transform
        except Exception:  # noqa: BLE001 - not localized: there's nowhere to put it
            self.get_logger().warning("crash not marked: the robot doesn't know where it is on the map")
            return False
        x, y = mark_at(t.translation.x, t.translation.y, yaw_of(t.rotation), direction)
        self.marks.append({'x': round(x, 3), 'y': round(y, 3), 'time': int(time.time()), 'jolt': round(jolt, 1)})
        save_marks(self.marks_path, self.marks)
        self.publish_list()
        self.publish_cloud()
        return True

    def on_remove(self, msg):
        kept = remove_marks(self.marks, msg.x, msg.y)
        if len(kept) == len(self.marks) or self.marks_path is None:
            return
        self.marks = kept
        save_marks(self.marks_path, kept)
        self.publish_list()
        self.publish_cloud()
        # The layer doesn't clear itself, so it has to be emptied; the sensors fill it back in.
        for client in self.clear_clients:
            if client.service_is_ready():
                client.call_async(ClearEntireCostmap.Request())

    # ---------- SLAM: its one service toggles, so only resume what this node paused ----------
    def pause_slam(self):
        if self.mode not in ('mapping', 'exploring'):
            return
        if self.slam_resume_at is None and self.slam_pause.service_is_ready():
            self.slam_pause.call_async(Pause.Request())
            self.get_logger().info('SLAM paused')
        if self.slam_pause.service_is_ready():
            self.slam_resume_at = time.monotonic() + SLAM_PAUSE_S

    def resume_slam(self):
        if self.slam_resume_at is not None and time.monotonic() >= self.slam_resume_at:
            self.slam_resume_at = None
            if self.slam_pause.service_is_ready():
                self.slam_pause.call_async(Pause.Request())
                self.get_logger().info('SLAM resumed')

    # ---------- housekeeping ----------
    def publish_cloud(self):
        pts = np.array(mark_points(self.marks), np.float32).reshape(-1, 3)
        m = PointCloud2()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), 'map'
        m.height, m.width = 1, len(pts)
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
        m.is_bigendian, m.point_step, m.row_step, m.is_dense = False, 12, 12 * len(pts), True
        m.data = pts.tobytes()
        self.cloud_pub.publish(m)

    def tick(self):
        # Single-threaded executor: destroying here can't race a take on another thread.
        if self.subs is not None and time.monotonic() - self.moved_at > WAKE_S:
            self.sleep()
        if self.cloud_pub.get_subscription_count():
            self.publish_cloud()


def main():
    rclpy.init()
    node = CrashMonitor()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
