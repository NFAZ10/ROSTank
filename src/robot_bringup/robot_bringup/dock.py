"""Find the home dock by its AprilTag, drive up to it and plug into the charger.

The dock carries a tag36h11 AprilTag (id tag_id, black square tag_size metres across) above
the dock's magnetic charge plug; the robot's plug is at the front, under the cameras.
/robot/go_dock (Trigger, while mapping or navigating):
  1. if this map remembers the dock (maps/<map>.dock.json), drive near it first
  2. look for the tag in the left camera, turning on the spot (one turn at most)
  3. drive (Nav2) to approach metres in front of the tag, facing it; remember where it is
  4. plug in (below)
/robot/plug_in (Trigger, any mode, tag in view): plug in from where the robot is:
  aligning   camera: steer onto the dock's centre line, square to the tag, until the tag is
             handoff_distance ahead. The dock's pose is averaged in the odom frame over the
             sightings, and steering runs on that + odometry (single frames are too noisy).
  repositioning  more than reposition_lateral off the line: turn to a point on the line
             reposition_distance out, drive there (backwards if it's behind), face the dock
  creeping   blind, creep_speed straight on, heading held on /odom (gyro), straight to
             /cmd_vel_safe: past drive_guard, which would stop 12 cm short of the dock.
             Stops when /charger/connected, at stop_range on /range/front, or after the
             expected distance (+ margin)
  settling   still; the charger should connect within settle_time
  backing    missed: reverse backoff metres and try again, max_tries in all
/robot/undock (Trigger): reverse backoff metres off the dock (through drive_guard).
/robot/learn_dock (Trigger, on the charger): measure where the tag and the front range are
  when plugged in and save it (geometry_file); used from then on. Re-run after moving or
  rebuilding the dock.
Auto-dock (go_dock by itself, not while on the charger; after one, none for auto_retry_s):
  battery     /battery/status low or critical, while mapping, exploring (stops the explorer
              first, the map keeps going) or navigating
  idle        navigating a map that knows its dock, nothing on /cmd_vel_safe for auto_idle_s
              (0 = off)
While docking: headlights dock_headlights (put back afterwards), /dock/active True (pico_io
shows the autonomous-driving flash).
/robot/check_dock (Trigger): look (no driving) for a few seconds; if the tag is in view,
  record the dock: for the map being navigated, or while mapping, in maps/.pending_dock.json,
  which nav_manager renames to <name>.dock.json when that map is saved.
Status (latched): /dock/status. The camera is only used while one of these runs.

Geometry: docked_distance (tag distance from base_link when plugged in), lateral_offset and
range_offset (tag distance - front range); learn_dock measures them. The creep never goes
closer than stop_margin past the docked front range.
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
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CompressedImage, Range
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from robot_bringup.drive_guard import matrix_of
from robot_bringup.stereo_depth import load_calib


MAP_MODES = ('mapping', 'exploring', 'navigating')  # a map frame exists
PLUG_STATES = ('aligning', 'repositioning', 'creeping', 'settling', 'backing')


def line_errors_2d(p, n, lateral_offset=0.0):
    """Tag centre p and face normal n (2D, base_link) -> (distance, lateral, heading).

    distance  from base_link to the tag plane, along the tag's normal
    lateral   how far the line is to the robot's left (+) or right (-), m
    heading   turn needed to face straight into the dock, rad (+ = left)
    lateral_offset shifts the line to the tag's left as seen by the robot (plug off-centre).
    """
    n = np.asarray(n, float) / max(np.linalg.norm(n), 1e-6)  # out of the tag's face
    u = np.array([n[1], -n[0]])  # along the tag face: the robot's left when square to it
    p = np.asarray(p, float)
    return float(p @ -n), float(p @ u - lateral_offset), math.atan2(-n[1], -n[0])


def line_errors(T_base_tag, lateral_offset=0.0):
    """As line_errors_2d, from the tag's 4x4 pose in base_link."""
    return line_errors_2d(T_base_tag[:2, 3], T_base_tag[:2, 2], lateral_offset)


def steer(lateral, heading, speed, gain=2.0, max_turn=0.4, softening=0.10, max_intercept=math.radians(12)):
    """Stanley-style steering onto the line: angular velocity, rad/s.

    The approach angle is capped at max_intercept off square, so the tag stays in view.
    """
    intercept = max(-max_intercept, min(max_intercept, math.atan2(lateral, softening + abs(speed))))
    return max(-max_turn, min(max_turn, gain * (heading + intercept)))


def rot(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


class DockEstimate:
    """The dock's tag centre and normal in the odom frame, averaged over sightings.

    The dock doesn't move, so averaging takes out the per-frame pose noise (a few degrees of
    normal at 0.6 m = cm of lateral error); a running average with a floor on the weight
    still follows the slow odometry drift.
    """

    def __init__(self, min_weight=0.2):
        self.p, self.n, self.count, self.min_weight = None, None, 0, min_weight

    def add(self, odom, p_base, n_base):
        x, y, yaw = odom
        R = rot(yaw)
        p = R @ np.asarray(p_base, float) + (x, y)
        n = R @ (np.asarray(n_base, float) / max(np.linalg.norm(n_base), 1e-6))
        self.count += 1
        a = max(1.0 / self.count, self.min_weight)
        if self.p is None:
            self.p, self.n = p, n
        else:
            self.p = (1 - a) * self.p + a * p
            self.n = (1 - a) * self.n + a * n
            self.n /= np.linalg.norm(self.n)

    def in_base(self, odom):
        """(p, n) in base_link for the robot at odom (x, y, yaw)."""
        x, y, yaw = odom
        R = rot(-yaw)
        return R @ (self.p - (x, y)), R @ self.n


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class Dock(Node):
    def __init__(self):
        super().__init__('dock')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.tag_id, self.tag_size = p('tag_id', 0), p('tag_size', 0.09)  # black square edge, m (user's print)
        self.approach = p('approach', 0.45)
        # Plugging in (see module doc). Distances are from base_link to the tag plane.
        self.range_offset = p('range_offset', 0.109)        # tag distance - front range
        self.docked_distance = p('docked_distance', 0.275)  # tag distance when plugged in
        self.handoff_min = p('handoff_distance', 0.36)      # camera -> blind creep here (at least)
        self.lateral_offset = p('lateral_offset', 0.0)      # dock line vs tag centre, m (+ = left)
        self.stop_margin = p('stop_margin', 0.015)
        self.geometry_file = os.path.expanduser(p('geometry_file', '~/robot_ws/maps/dock_geometry.json'))
        self.load_geometry()
        self.lateral_tol = p('lateral_tol', 0.010)
        self.heading_tol = math.radians(p('heading_tol_deg', 3.0))
        self.align_speed, self.creep_speed = p('align_speed', 0.06), p('creep_speed', 0.04)
        self.creep_margin = p('creep_margin', 0.04)         # past the expected distance
        self.settle_time = p('settle_time', 3.0)
        self.backoff, self.max_tries = p('backoff', 0.30), p('max_tries', 3)
        self.reposition_lateral = p('reposition_lateral', 0.045)
        self.reposition_distance = p('reposition_distance', 0.55)  # tag distance of the waypoint
        self.auto_battery = p('auto_dock_battery', True)
        self.auto_idle_s = p('auto_idle_s', 300.0)
        self.auto_retry_s = p('auto_retry_s', 600.0)
        self.dock_headlights = p('dock_headlights', 'on')  # '' = leave them alone
        self.map_dir = os.path.expanduser(p('map_dir', '~/robot_ws/maps'))
        calib = load_calib(os.path.join(get_package_share_directory('robot_bringup'),
                                        'config', 'calibration', 'left.yaml'))
        self.calib_K, self.calib_D, self.calib_size = calib['K'], calib['D'], calib['size']
        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/dock/status', latched)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.direct_pub = self.create_publisher(Twist, '/cmd_vel_safe', 10)  # past drive_guard
        self.mode, self.nav_map = 'idle', ''
        self.mode_since = time.monotonic()
        self.create_subscription(String, '/robot/nav_mode', self.on_mode, latched)
        self.create_subscription(String, '/robot/nav_map', lambda m: setattr(self, 'nav_map', m.data), latched)
        self.charger = False
        self.create_subscription(Bool, '/charger/connected', lambda m: setattr(self, 'charger', m.data), latched)
        self.front_range = (math.inf, 0.0)
        self.create_subscription(Range, '/range/front', self.on_range, qos_profile_sensor_data)
        self.odom = None  # (x, y, yaw)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)
        self.battery = ''
        self.create_subscription(String, '/battery/status', lambda m: setattr(self, 'battery', m.data), latched)
        self.last_motion = time.monotonic()
        self.create_subscription(Twist, '/cmd_vel_safe', self.on_cmd, 10)
        self.headlights, self.saved_headlights = None, None
        self.create_subscription(String, '/lights/status', self.on_lights, latched)
        self.headlights_pub = self.create_publisher(String, '/lights/headlights', 10)
        self.active_pub = self.create_publisher(Bool, '/dock/active', latched)
        self.active = None
        self.auto_block_until, self.auto_pending = 0.0, ''
        self.stop_exploring = self.create_client(Trigger, '/robot/stop_exploring')
        self.nav = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.create_service(Trigger, '/robot/go_dock', self.on_go_dock)
        self.create_service(Trigger, '/robot/check_dock', self.on_check_dock)
        self.create_service(Trigger, '/robot/plug_in', self.on_plug_in)
        self.create_service(Trigger, '/robot/undock', self.on_undock)
        self.create_service(Trigger, '/robot/learn_dock', self.on_learn_dock)
        self.learn_samples = []

        self.state, self.image_sub, self.frame = 'idle', None, None
        self.frame_time, self.used_frame = 0.0, None
        self.camera_waiting = False
        self.goal_handle, self.goal_done, self.search_start = None, False, 0.0
        self.tries, self.state_start, self.start_xy = 0, 0.0, None
        self.retrying, self.drive_dist, self.drive_back, self.drive_yaw = False, 0.0, False, 0.0
        self.hold_yaw, self.creep_limit, self.last_tag = 0.0, 0.0, 0.0
        self.estimate = DockEstimate()
        self.waypoint, self.phase, self.phase_start = None, '', 0.0
        self.say('')
        self.create_timer(0.1, self.tick)
        self.create_timer(1.0, self.auto_dock)

    def say(self, text):
        self.status_pub.publish(String(data=text))
        if text:
            self.get_logger().info(text)

    # ---------- geometry ----------
    def load_geometry(self):
        try:
            with open(self.geometry_file) as f:
                g = json.load(f)
            self.docked_distance, self.lateral_offset = g['docked_distance'], g['lateral_offset']
            self.range_offset = g['range_offset']
            self.get_logger().info(f'dock geometry from {self.geometry_file}: {g}')
        except (OSError, ValueError, KeyError):
            pass
        self.handoff_distance = max(self.handoff_min, self.docked_distance + 0.08)
        self.stop_range = self.docked_distance - self.range_offset - self.stop_margin

    def on_learn_dock(self, request, response):
        if self.busy(response):
            return response
        if not self.charger:
            response.success, response.message = False, 'park it on the charger first (the charger must be connected)'
            return response
        self.camera(True)
        self.learn_samples = []
        self.set_state('learning')
        self.say('measuring the dock')
        response.success, response.message = True, 'learning the dock (keep it still on the charger)'
        return response

    def tick_learn(self, elapsed):
        T = self.fresh_tag()
        if T is not None and not isinstance(T, str):
            rng, rng_t = self.front_range
            if time.monotonic() - rng_t < 0.5 and math.isfinite(rng):
                d, lat, hdg = line_errors(T)
                self.learn_samples.append((d, lat, d - rng))
        if len(self.learn_samples) >= 15:
            a = np.median(np.array(self.learn_samples), axis=0)
            g = {'docked_distance': round(float(a[0]), 4), 'lateral_offset': round(float(a[1]), 4),
                 'range_offset': round(float(a[2]), 4)}
            with open(self.geometry_file, 'w') as f:
                json.dump(g, f)
            self.load_geometry()
            return self.finish(f'dock learned: tag {g["docked_distance"] * 100:.1f} cm, '
                               f'{g["lateral_offset"] * 100:+.1f} cm sideways, '
                               f'front range {(g["docked_distance"] - g["range_offset"]) * 100:.1f} cm')
        if elapsed > 45.0:
            self.finish("couldn't measure the dock: the whole tag must be in view while plugged in")

    def on_mode(self, msg):
        if msg.data != self.mode:
            self.mode, self.mode_since = msg.data, time.monotonic()

    def on_cmd(self, msg):
        if abs(msg.linear.x) > 0.005 or abs(msg.angular.z) > 0.02:
            self.last_motion = time.monotonic()

    def on_lights(self, msg):
        try:
            self.headlights = json.loads(msg.data).get('headlights')
        except ValueError:
            pass

    def on_range(self, msg):
        self.front_range = (msg.range, time.monotonic())

    def on_odom(self, msg):
        p = msg.pose.pose
        self.odom = (p.position.x, p.position.y, yaw_of(p.orientation))

    def travelled(self):
        if self.odom is None or self.start_xy is None:
            return 0.0
        return math.hypot(self.odom[0] - self.start_xy[0], self.odom[1] - self.start_xy[1])

    # ---------- camera ----------
    def camera(self, on):
        if on and self.image_sub is None:
            self.image_sub = self.create_subscription(CompressedImage, '/left/preview/image/compressed',
                                                      self.on_frame, qos_profile_sensor_data)
            self.camera_waiting = True
            if self.dock_headlights and self.headlights is not None and self.headlights != self.dock_headlights:
                self.saved_headlights = self.headlights  # lit, the tag reads further and in the dark
                self.headlights_pub.publish(String(data=self.dock_headlights))
        elif not on and self.image_sub is not None:
            self.destroy_subscription(self.image_sub)
            self.image_sub, self.frame = None, None
            if self.saved_headlights is not None:
                self.headlights_pub.publish(String(data=self.saved_headlights))
                self.saved_headlights = None

    def on_frame(self, msg):
        self.frame, self.frame_time = msg, time.monotonic()

    def tag_in_camera(self):
        """Tag pose as a 4x4 in the left optical frame from the newest frame, or None."""
        m = self.frame
        if m is None:
            return None
        img = cv2.imdecode(np.frombuffer(m.data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        corners, ids, _ = self.detector.detectMarkers(img)
        if ids is None or self.tag_id not in ids.ravel():
            return None
        c = corners[list(ids.ravel()).index(self.tag_id)].reshape(4, 2)
        h_, w_ = img.shape[:2]
        S = np.diag([w_ / self.calib_size[0], h_ / self.calib_size[1], 1.0])  # calib is full-res
        h = self.tag_size / 2
        obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)  # OpenCV corner order
        ok, rv, tv = cv2.solvePnP(obj, c, S @ self.calib_K, self.calib_D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        T_cam_tag = np.eye(4)
        T_cam_tag[:3, :3], T_cam_tag[:3, 3] = cv2.Rodrigues(rv)[0], tv.ravel()
        return T_cam_tag

    def find_tag(self, frame='map'):
        """Tag pose as a 4x4 in frame (map or base_link), or None."""
        T_cam_tag = self.tag_in_camera()
        if T_cam_tag is None:
            return None
        try:
            T = matrix_of(self.tf.lookup_transform(frame, 'stereo_camera_left_optical', Time()).transform)
        except Exception:  # noqa: BLE001 - not localized
            return None
        return T @ T_cam_tag

    def fresh_tag(self):
        """Tag in base_link from a frame not used yet and under 0.5 s old; 'old' if none new."""
        if self.frame is None or self.frame is self.used_frame or time.monotonic() - self.frame_time > 0.5:
            return 'old'
        self.used_frame = self.frame
        return self.find_tag('base_link')

    # ---------- driving ----------
    def drive(self, v, w, direct=False):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        (self.direct_pub if direct else self.cmd_pub).publish(t)

    def stop(self):
        self.drive(0.0, 0.0)
        self.drive(0.0, 0.0, direct=True)

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

    # ---------- services ----------
    def busy(self, response):
        if self.state != 'idle':
            response.success, response.message = False, 'busy with the dock already'
            return True
        return False

    def on_check_dock(self, request, response):
        if self.mode not in MAP_MODES:
            response.success, response.message = False, 'start mapping or navigation first'
            return response
        if self.busy(response):
            return response
        self.camera(True)
        self.state, self.search_start = 'checking', time.monotonic()
        self.say('checking for the dock tag')
        response.success, response.message = True, 'checking for the dock (not moving)'
        return response

    def on_go_dock(self, request, response):
        if self.busy(response):
            return response
        response.success, response.message = self.start_go_dock()
        return response

    def start_go_dock(self, wait=2.0):
        """(started, message)."""
        if self.mode not in MAP_MODES:
            return False, 'start mapping or navigation first'
        if self.charger:
            return True, 'already on the charger'
        if not self.nav.wait_for_server(timeout_sec=wait):
            return False, 'navigation is still starting'
        self.camera(True)
        self.tries = 0
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
        return True, 'going to the dock'

    def on_plug_in(self, request, response):
        if self.busy(response):
            return response
        if self.charger:
            response.success, response.message = True, 'already on the charger'
            return response
        self.camera(True)
        self.tries = 0
        self.start_plug_in()
        response.success, response.message = True, 'plugging in (the dock tag must be in view)'
        return response

    def on_undock(self, request, response):
        if self.busy(response):
            return response
        if self.odom is None:
            response.success, response.message = False, 'no /odom yet'
            return response
        self.set_state('undocking')
        self.say('backing off the dock')
        response.success, response.message = True, 'undocking'
        return response

    # ---------- state machine ----------
    def set_state(self, state):
        self.state, self.state_start = state, time.monotonic()
        self.start_xy = self.odom[:2] if self.odom else None
        self.last_tag = time.monotonic()

    def start_plug_in(self, keep_estimate=False):
        self.tries += 1
        if not keep_estimate:
            self.estimate = DockEstimate()
        self.set_state('aligning')
        self.say(f'lining up with the dock (try {self.tries} of {self.max_tries})')

    def retry(self, why):
        self.stop()
        if self.tries >= self.max_tries:
            return self.finish(f"couldn't plug in after {self.tries} tries: {why}")
        if self.estimate.count >= 3 and self.odom is not None:  # back onto the line itself
            self.say(f'{why}: going back onto the dock line to try again')
            return self.start_reposition(retry=True)
        self.set_state('backing')
        self.say(f'{why}: backing off to try again')

    def start_reposition(self, retry=False):
        """Waypoint on the dock line, reposition_distance out (odom frame), then turn/drive/face."""
        out = self.reposition_distance
        self.waypoint = self.estimate.p + self.estimate.n * out + self.lateral_offset * np.array(
            [self.estimate.n[1], -self.estimate.n[0]]) * -1.0
        self.retrying = retry
        self.set_state('repositioning')
        self.set_phase('turn')

    def set_phase(self, phase):
        self.phase, self.phase_start = phase, time.monotonic()
        self.start_xy = self.odom[:2] if self.odom else None

    def finish(self, text):
        self.stop()
        self.camera(False)
        self.state = 'idle'
        self.say(text)

    def auto_dock(self):
        """Battery low, or idle on a map with a dock: go home and plug in."""
        now = time.monotonic()
        if self.state != 'idle' or self.charger or now < self.auto_block_until:
            self.auto_pending = ''
            return
        reason = self.auto_pending
        if not reason:
            if self.auto_battery and self.battery.startswith(('low', 'critical')) and self.mode in MAP_MODES:
                reason = f'battery {self.battery.split(":")[0]}'
            elif (self.auto_idle_s > 0 and self.mode == 'navigating' and self.dock_file()
                  and os.path.exists(self.dock_file())
                  and now - max(self.last_motion, self.mode_since) > self.auto_idle_s):
                reason = f'idle {self.auto_idle_s / 60:.0f} min'
        if not reason:
            return
        if self.mode == 'exploring':  # the explorer would keep driving: stop it, the map stays
            if self.stop_exploring.service_is_ready() and not self.auto_pending:
                self.stop_exploring.call_async(Trigger.Request())
            self.auto_pending = reason
            return
        self.auto_pending = ''
        self.auto_block_until = now + self.auto_retry_s  # one go; a failure doesn't loop
        ok, msg = self.start_go_dock(wait=0.5)
        if ok:
            self.say(f'auto-dock ({reason}): {msg}')
        else:
            self.get_logger().warn(f'auto-dock ({reason}) not started: {msg}')

    def publish_active(self):
        active = self.state not in ('idle', 'checking', 'learning')
        if active != self.active:
            self.active = active
            self.active_pub.publish(Bool(data=active))

    def tick(self):
        self.publish_active()
        if self.state == 'idle':
            return
        if self.state == 'learning':
            return self.tick_learn(time.monotonic() - self.state_start)
        if self.state in PLUG_STATES or self.state == 'undocking':
            return self.tick_plug()
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
            if self.goal_done != 4:
                return self.finish("couldn't reach the dock (blocked?)")
            self.start_plug_in()

    def tick_plug(self):
        now = time.monotonic()
        elapsed = now - self.state_start
        if self.charger and self.state in ('aligning', 'creeping', 'settling'):
            return self.finish('docked: charging')
        if self.state == 'aligning':
            if self.frame is None:  # the camera starts on demand
                if elapsed > 45.0:
                    self.finish("the camera didn't start: check robot logs cameras")
                return
            if self.odom is None:
                return self.finish('no /odom')
            T = self.fresh_tag()
            if T is not None and not isinstance(T, str):
                self.last_tag = now
                self.estimate.add(self.odom, T[:2, 3], T[:2, 2])
            if now - self.last_tag > 4.0:
                return self.finish('lost sight of the dock tag')
            if self.estimate.count < 3:  # a few sightings before moving on them
                return self.stop()
            if elapsed > 45.0:
                return self.retry('lining up took too long')
            distance, lateral, heading = line_errors_2d(*self.estimate.in_base(self.odom), self.lateral_offset)
            if abs(lateral) > self.reposition_lateral and self.estimate.count >= 5:
                self.say(f'{lateral * 100:+.0f} cm off the dock line: driving onto it first')
                return self.start_reposition()
            if distance > self.handoff_distance + 0.01:
                v = self.align_speed if distance > self.handoff_distance + 0.08 else self.align_speed / 2
                w = steer(lateral, heading, v)
                self.get_logger().info(f'align d {distance:.3f} lat {lateral * 100:+.1f} cm '
                                       f'hdg {math.degrees(heading):+.1f} deg -> v {v:.2f} w {w:+.2f} '
                                       f'({self.estimate.count} sightings)', throttle_duration_sec=0.4)
                return self.drive(v, w)
            self.stop()
            if abs(lateral) > self.lateral_tol:
                return self.retry(f'{lateral * 100:+.1f} cm off the dock line')
            if abs(heading) > self.heading_tol:  # square up on the spot
                return self.drive(0.0, max(-0.3, min(0.3, 1.5 * heading)))
            self.hold_yaw = wrap(self.odom[2] + heading)
            self.creep_limit = distance - self.docked_distance + self.creep_margin
            self.set_state('creeping')
            return self.say(f'creeping in ({lateral * 100:+.1f} cm, {math.degrees(heading):+.1f} deg)')
        if self.state == 'repositioning':
            return self.tick_reposition(now)
        if self.state == 'creeping':
            rng, rng_t = self.front_range
            if now - rng_t < 0.5 and rng <= self.stop_range:
                self.stop()
                return self.set_state('settling')
            if self.travelled() >= self.creep_limit or elapsed > self.creep_limit / self.creep_speed * 2.5 + 2:
                self.stop()
                return self.set_state('settling')
            w = max(-0.3, min(0.3, 2.0 * wrap(self.hold_yaw - self.odom[2])))
            return self.drive(self.creep_speed, w, direct=True)
        if self.state == 'settling':
            if elapsed > self.settle_time:
                self.retry('touched the dock but no charge')
            return
        if self.state in ('backing', 'undocking'):
            if self.travelled() >= self.backoff:
                self.stop()
                if self.state == 'undocking':
                    return self.finish('off the dock')
                return self.start_plug_in()
            if elapsed > self.backoff / 0.06 + 5:
                return self.finish("couldn't back away from the dock (blocked behind?)")
            return self.drive(-0.08, 0.0)

    def tick_reposition(self, now):
        x, y, yaw = self.odom
        to = self.waypoint - (x, y)
        dist = float(np.hypot(*to))
        backwards = float(to @ (math.cos(yaw), math.sin(yaw))) < 0
        if now - self.state_start > 60.0:
            return self.retry('repositioning took too long')
        if self.phase == 'turn':
            if dist < 0.03:
                return self.set_phase('face')
            target = math.atan2(to[1], to[0]) + (math.pi if backwards else 0.0)
            err = wrap(target - yaw)
            if abs(err) < math.radians(3):
                self.stop()
                self.drive_dist, self.drive_back, self.drive_yaw = dist, backwards, target
                return self.set_phase('drive')
            return self.drive(0.0, max(-0.4, min(0.4, 1.5 * err)) if abs(err) > 0.1 else math.copysign(0.15, err))
        if self.phase == 'drive':
            if self.travelled() >= self.drive_dist:
                self.stop()
                return self.set_phase('face')
            w = max(-0.3, min(0.3, 2.0 * wrap(self.drive_yaw - yaw)))
            return self.drive(-0.06 if self.drive_back else 0.06, w)
        if self.phase == 'face':
            target = math.atan2(-self.estimate.n[1], -self.estimate.n[0])
            err = wrap(target - yaw)
            if abs(err) < math.radians(2):
                self.stop()
                self.last_tag = now  # give the camera a moment to find the tag again
                if self.retrying:
                    return self.start_plug_in(keep_estimate=True)
                self.state, self.state_start = 'aligning', now
                return self.say('on the dock line: lining up')
            return self.drive(0.0, max(-0.4, min(0.4, 1.5 * err)) if abs(err) > 0.1 else math.copysign(0.15, err))

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
