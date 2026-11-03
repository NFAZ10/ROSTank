"""Find the home dock by its AprilTag and drive up to it.

The dock carries a tag36h11 AprilTag (id tag_id, black square tag_size metres across).
/robot/go_dock (Trigger, while mapping or navigating):
  1. if this map remembers the dock (maps/<map>.dock.json), drive near it first
  2. look for the tag in the left camera, turning on the spot (one turn at most)
  3. drive to approach metres in front of the tag, facing it; remember where it is
/robot/check_dock (Trigger): look (no driving) for a few seconds; if the tag is in view,
  record the dock: for the map being navigated, or while mapping, in maps/.pending_dock.json,
  which nav_manager renames to <name>.dock.json when that map is saved.
Status (latched): /dock/status. The camera is only used during a go_dock or check_dock.
"""
import json
import math
import os
import time

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from robot_bringup.drive_guard import matrix_of
from robot_bringup.stereo_depth import load_calib


MAP_MODES = ('mapping', 'exploring', 'navigating')  # a map frame exists


class Dock(Node):
    def __init__(self):
        super().__init__('dock')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.tag_id, self.tag_size = p('tag_id', 0), p('tag_size', 0.09)  # black square edge, m (user's print)
        self.approach = p('approach', 0.45)
        self.map_dir = os.path.expanduser(p('map_dir', '~/robot_ws/maps'))
        calib = load_calib(os.path.join(get_package_share_directory('robot_bringup'),
                                        'config', 'calibration', 'left.yaml'))
        self.calib_K, self.calib_D, self.calib_size = calib['K'], calib['D'], calib['size']
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/dock/status', latched)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.mode, self.nav_map = 'idle', ''
        self.create_subscription(String, '/robot/nav_mode', lambda m: setattr(self, 'mode', m.data), latched)
        self.create_subscription(String, '/robot/nav_map', lambda m: setattr(self, 'nav_map', m.data), latched)
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.create_service(Trigger, '/robot/go_dock', self.on_go_dock)
        self.create_service(Trigger, '/robot/check_dock', self.on_check_dock)

        self.state, self.image_sub, self.frame = 'idle', None, None
        self.camera_waiting = False
        self.goal_handle, self.goal_done, self.search_start = None, False, 0.0
        self.say('')
        self.create_timer(0.2, self.tick)

    def say(self, text):
        self.status_pub.publish(String(data=text))
        if text:
            self.get_logger().info(text)

    # ---------- camera ----------
    def camera(self, on):
        if on and self.image_sub is None:
            self.image_sub = self.create_subscription(Image, '/left/preview/image',
                                                      lambda m: setattr(self, 'frame', m), qos_profile_sensor_data)
            self.camera_waiting = True
        elif not on and self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            self.image_sub, self.frame = None, None

    def find_tag(self):
        """Tag pose as a 4x4 in the map frame, or None."""
        m = self.frame
        if m is None:
            return None
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * 3].reshape(m.height, m.width, 3)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if m.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or self.tag_id not in ids.ravel():
            return None
        c = corners[list(ids.ravel()).index(self.tag_id)].reshape(4, 2)
        S = np.diag([m.width / self.calib_size[0], m.height / self.calib_size[1], 1.0])  # calib is full-res
        h = self.tag_size / 2
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)  # OpenCV corner order
        ok, rv, tv = cv2.solvePnP(obj, c, S @ self.calib_K, self.calib_D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3], T_cam_tag[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
        try:
            T_map_cam = matrix_of(self.tf.lookup_transform('map', 'stereo_camera_left_optical', Time()).transform)
        except Exception:  # noqa: BLE001 - not localized
            return None
        return T_map_cam @ T_cam_tag

    # ---------- driving ----------
    def go_to(self, x, y, yaw, label):
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = 'map'
        g.pose.header.stamp = self.get_clock().now().to_msg()
        g.pose.pose.position.x, g.pose.pose.position.y = float(x), float(y)
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.goal_done, self.goal_handle = False, None
        self.nav.send_goal_async(g).add_done_callback(self.on_accepted)
        self.say(label)

    def on_accepted(self, future):
        handle = future.result()
        if not handle.accepted:
            self.goal_done = 'rejected'
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(lambda f: setattr(self, 'goal_done', f.result().status))

    def dock_file(self):
        """Where this map's dock is kept: the named map, or pending until a new map is saved."""
        if self.mode == 'navigating' and self.nav_map:
            return os.path.join(self.map_dir, f'{self.nav_map}.dock.json')
        if self.mode in ('mapping', 'exploring'):
            return os.path.join(self.map_dir, '.pending_dock.json')
        return None

    def parking_spot(self, T):
        """Tag pose (map) -> (x, y, yaw) approach metres in front of it, facing it; also saved."""
        n = T[:2, 2] / max(np.linalg.norm(T[:2, 2]), 1e-6)  # tag z axis points out of its face
        gx, gy = T[:2, 3] + n * self.approach
        yaw = math.atan2(-n[1], -n[0])
        if self.dock_file():
            with open(self.dock_file(), 'w') as f:
                json.dump({'x': round(float(gx), 3), 'y': round(float(gy), 3), 'yaw': round(yaw, 4)}, f)
        return gx, gy, yaw

    def on_check_dock(self, request, response):
        if self.mode not in MAP_MODES:
            response.success, response.message = False, 'start mapping or navigation first'
            return response
        if self.state != 'idle':
            response.success, response.message = False, 'busy with the dock already'
            return response
        self.camera(True)
        self.state, self.search_start = 'checking', time.monotonic()
        self.say('checking for the dock tag')
        response.success, response.message = True, 'checking for the dock (not moving)'
        return response

    def on_go_dock(self, request, response):
        if self.mode not in MAP_MODES:
            response.success, response.message = False, 'start mapping or navigation first'
            return response
        if not self.nav.wait_for_server(timeout_sec=2.0):
            response.success, response.message = False, 'navigation is still starting'
            return response
        self.camera(True)
        remembered = None
        if self.dock_file() and os.path.exists(self.dock_file()):
            with open(self.dock_file()) as f:
                remembered = json.load(f)
        if remembered:
            self.state = 'to_remembered'
            self.go_to(remembered['x'], remembered['y'], remembered['yaw'], 'driving to where the dock was')
        else:
            self.state, self.search_start = 'searching', time.monotonic()
            self.say('looking for the dock tag')
        response.success, response.message = True, 'going to the dock'
        return response

    def finish(self, text):
        self.cmd_pub.publish(Twist())
        self.camera(False)
        self.state = 'idle'
        self.say(text)

    def tick(self):
        if self.state == 'idle':
            return
        if self.mode not in MAP_MODES:
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            return self.finish('stopped')
        if self.state == 'to_remembered':
            if self.goal_done:
                self.state, self.search_start = 'searching', time.monotonic()
            return
        if self.state in ('checking', 'searching') and self.camera_waiting:
            # The camera starts on demand (camera_manager.py): the search clock begins with its first frame.
            if self.frame is not None:
                self.camera_waiting, self.search_start = False, time.monotonic()
            elif time.monotonic() - self.search_start > 45.0:
                return self.finish("the camera didn't start: check robot logs cameras")
            else:
                return
        if self.state == 'checking':
            T = self.find_tag()
            if T is not None:
                self.parking_spot(T)
                where = 'saved for this map' if self.mode == 'navigating' else 'it will be saved with the map'
                return self.finish(f'dock found {np.hypot(*(T[:2, 3] - self.robot_xy())):.1f} m away: {where}')
            if time.monotonic() - self.search_start > 4.0:
                return self.finish('no dock tag in view: point the camera at it and check again')
            return
        if self.state == 'searching':
            T = self.find_tag()
            if T is not None:
                self.cmd_pub.publish(Twist())
                gx, gy, yaw = self.parking_spot(T)
                self.state = 'approaching'
                self.go_to(gx, gy, yaw, f'dock tag seen {np.hypot(*(T[:2, 3] - self.robot_xy())):.1f} m away: driving up')
            elif time.monotonic() - self.search_start > 2 * math.pi / 0.4:
                self.finish("couldn't see the dock tag: point the camera at it and try again")
            else:
                t = Twist()
                t.angular.z = 0.4  # slow: the 640x480 preview must stay sharp enough to decode
                self.cmd_pub.publish(t)
            return
        if self.state == 'approaching' and self.goal_done:
            self.finish('at the dock' if self.goal_done == 4 else "couldn't reach the dock (blocked?)")

    def robot_xy(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time()).transform.translation
            return np.array([t.x, t.y])
        except Exception:  # noqa: BLE001
            return np.zeros(2)


def main():
    rclpy.init()
    node = Dock()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
