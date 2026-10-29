"""Drive guard: every drive command passes through here before the motors.

  /cmd_vel (web UI, CLI, teleop, Nav2)  ->  drive_guard  ->  /cmd_vel_safe  ->  pico_bridge

Looks along the direction of travel in a corridor the robot's width plus a margin: full
speed beyond slow_dist, slowing down closer in, stopped at stop_dist (turning on the spot
stays allowed, so the robot can be steered away). Obstacles come from the lidar (all
around) and stereo depth (ahead, 10-50 cm high: things under the lidar's plane).

Sensors are only subscribed while driving and for wake_timeout after, so the lidar motor
(lidar_standby.py) and stereo depth (stereo_depth.py) still idle when nothing moves. While
the lidar spins up there is no fresh scan: until then only turning is let through.
Why it is holding back is latched on /drive_guard/status.

/wheel/cmd (per-track bench test, tracks off) does not pass through here.
"""
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener


def matrix_of(t):
    """4x4 transform from a geometry_msgs Transform."""
    q, p = t.rotation, t.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, [p.x, p.y, p.z]
    return M


class DriveGuard(Node):
    def __init__(self):
        super().__init__('drive_guard')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.half_length = p('half_length', 0.16)   # footprint (= Nav2 footprint), m
        self.half_width = p('half_width', 0.125)
        self.margin = p('side_margin', 0.04)       # corridor = footprint width + this each side
        self.stop_dist = p('stop_dist', 0.12)      # clearance at which forward/back motion stops
        self.slow_dist = p('slow_dist', 0.5)       # clearance where slowing down starts
        self.min_speed = p('min_speed', 0.05)      # slowed commands never go below this (m/s)
        self.wake_timeout = p('wake_timeout', 10.0)  # s after the last command before sensors idle
        self.use_depth = p('use_depth', True)
        self.depth_height = (p('depth_min_height', 0.10), p('depth_max_height', 0.5))
        self.depth_range = p('depth_max_range', 1.8)

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_safe', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/drive_guard/status', latched)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd, 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.frames = {}  # sensor frame -> 4x4 into base_link (both mounts are fixed)

        self.cmd, self.cmd_time, self.sent_zero = None, 0.0, True
        self.subs = None
        self.scan_pts, self.scan_time = np.zeros((0, 2)), 0.0
        self.depth_pts, self.depth_time = np.zeros((0, 2)), 0.0
        self.status = None
        self.set_status('idle')
        self.create_timer(0.05, self.step)
        self.create_timer(1.0, self.manage)

    # ---------- sensors on demand ----------
    def wake(self):
        if self.subs is None:
            self.subs = [self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)]
            if self.use_depth:
                self.subs.append(self.create_subscription(PointCloud2, '/depth/points', self.on_depth,
                                                          qos_profile_sensor_data))

    def manage(self):
        # Single-threaded executor: destroying here can't race a take on another thread.
        if self.subs is not None and time.monotonic() - self.cmd_time > self.wake_timeout:
            for sub in self.subs:
                self.destroy_subscription(sub)
            self.subs = None
            self.scan_time = self.depth_time = 0.0
            self.set_status('idle')

    def to_base(self, frame):
        if frame not in self.frames:
            try:
                self.frames[frame] = matrix_of(self.tf.lookup_transform('base_link', frame, Time()).transform)
            except Exception:  # noqa: BLE001 - TF not there yet; next message
                return None
        return self.frames[frame]

    def on_scan(self, msg):
        M = self.to_base(msg.header.frame_id)
        if M is None:
            return
        r = np.asarray(msg.ranges, np.float32)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        pts = np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), np.zeros(ok.sum()), np.ones(ok.sum())])
        self.scan_pts, self.scan_time = (M @ pts)[:2].T, time.monotonic()

    def on_depth(self, msg):
        M = self.to_base(msg.header.frame_id)
        if M is None or msg.width * msg.height == 0:
            return
        xyz = np.frombuffer(msg.data, np.float32).reshape(-1, msg.point_step // 4)[:, :3]
        xyz = xyz[xyz[:, 2] < self.depth_range]  # optical z = distance
        pts = M @ np.vstack([xyz.T, np.ones(len(xyz))])
        keep = (pts[2] > self.depth_height[0]) & (pts[2] < self.depth_height[1])
        self.depth_pts, self.depth_time = pts[:2, keep].T, time.monotonic()

    # ---------- guarding ----------
    def set_status(self, text):
        if text != self.status:
            self.status = text
            self.status_pub.publish(String(data=text))

    def on_cmd(self, msg):
        self.cmd, self.cmd_time = msg, time.monotonic()
        self.wake()
        self.step()  # react now, not on the next tick

    def clearance(self, direction):
        """Free distance from the front (direction 1) or back (-1) edge to the nearest obstacle."""
        now = time.monotonic()
        sets = [self.scan_pts]
        if direction > 0 and now - self.depth_time < 1.0:
            sets.append(self.depth_pts)  # cameras face forward
        pts = np.vstack(sets)
        x = pts[:, 0] * direction
        ahead = (np.abs(pts[:, 1]) < self.half_width + self.margin) & (x > self.half_length)
        return float(x[ahead].min() - self.half_length) if ahead.any() else math.inf

    def step(self):
        now = time.monotonic()
        if self.cmd is None or now - self.cmd_time > 0.5:
            if not self.sent_zero:
                self.cmd_pub.publish(Twist())
                self.sent_zero = True
            if self.subs is not None:
                self.set_status('ready')
            return
        out = Twist()
        out.angular.z = self.cmd.angular.z
        v = self.cmd.linear.x
        if abs(v) > 1e-3:
            if now - self.scan_time > 0.5:
                v = 0.0
                self.set_status('waiting for lidar (turning only)')
            else:
                gap = self.clearance(1 if v > 0 else -1)
                where = 'ahead' if v > 0 else 'behind'
                if gap <= self.stop_dist:
                    v = 0.0
                    self.set_status(f'stopped: obstacle {max(gap, 0):.2f} m {where}')
                elif gap < self.slow_dist:
                    scale = (gap - self.stop_dist) / (self.slow_dist - self.stop_dist)
                    v = math.copysign(max(abs(v) * scale, min(abs(v), self.min_speed)), v)
                    self.set_status(f'slowing: obstacle {gap:.2f} m {where}')
                else:
                    self.set_status('clear')
        out.linear.x = v
        self.cmd_pub.publish(out)
        self.sent_zero = False


def main():
    rclpy.init()
    node = DriveGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
