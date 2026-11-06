"""Robot web UI: http://<robot-ip>:8080 (page in robot_bringup/web/index.html).

Map + lidar view, live camera (MJPEG from the JPEG previews), mapping/navigation start-stop (via nav_manager), hold-to-drive,
nav goals, and drive calibration (direction test, 1 m straight, 360 degree spin).
/encoders: live hall-encoder counts and speeds per track, and hold-to-spin one track (web/encoders.html).
/imu: live gyro and accelerometer charts from the IMU (web/imu.html).
/calibrate: stereo checkerboard calibration (live detection, auto-capture, OpenCV calibration,
writes config/calibration/left.yaml + right.yaml; see stereo_calib.py).
/mapedit: hand-edit a saved map (paint walls, free space, unknown; original kept as <name>.orig.pgm)
          and its no-go zones (<name>.keepout.pgm/.yaml: Nav2's keepout filter mask, see navigation.launch.py).
Saved maps: named saves, a list with thumbnails, navigate on / delete (files in map_dir, run by nav_manager).
Goals go to Nav2 as a NavigateToPose action, so the page sees accepted / distance left / reached / failed.
Calibration results are applied live to pico_bridge and written to drive.yaml.
Runs as its own service (robot-web.service) so it can start/stop the robot stack
(robot.service) through passwordless sudo.

While the stack is down it also holds the lidar's serial port: an RPLIDAR A1 runs its
motor whenever no process holds DTR, so it would otherwise spin with the stack stopped.

/scan and the camera previews are only subscribed while a browser is using them, so the lidar motor still
stops when nobody is looking (lidar_standby.py).

Idle auto-stop: the stack is stopped after idle_timeout seconds (default 30 min) with no track
motion (wheel/joint_states), no mapping/navigation, no request from this page and no Foxglove or rosbridge client connected.
"""
import base64
import collections
import glob
import io
import json
import math
import os
import re
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
import serial
import yaml
from ament_index_python.packages import get_package_share_directory
from action_msgs.msg import GoalStatus
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import LoadMap
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, Point
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import (BatteryState, CompressedImage, Image, Imu, JointState, LaserScan, Range,
                             RelativeHumidity, Temperature)
from std_msgs.msg import Bool, Float32, Float64MultiArray, String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformListener

from robot_bringup.stereo_calib import NOVELTY, SIDES, StereoSession, find_corners, sample_params

CALIB_PARAMS = ['wheel_radius', 'track_width', 'invert_left', 'invert_right', 'swap_sides']
NAV_COMMANDS = ['start_mapping', 'start_exploring', 'save_map', 'start_navigation', 'stop', 'localize', 'find_myself', 'go_dock', 'check_dock', 'continue_mapping', 'plug_in', 'undock', 'learn_dock']
MAP_NAME_RE = re.compile(r'^[A-Za-z0-9_-]{1,40}$')  # same rule as nav_manager
MAX_V, MAX_W = 0.4, 1.5
STACK_UNIT = 'robot.service'
CAMERAS = ['left', 'right', 'depth']
SCAN_MAX_RANGE = 6.0  # m; matches SLAM max_laser_range / AMCL laser_max_range
CAMERA_TOPICS = {'left': '/left/preview/image/compressed', 'right': '/right/preview/image/compressed',
                 'depth': '/depth/preview/compressed'}  # depth: colour-coded, from stereo_depth.py
STACK_ACTIONS = ['start', 'stop', 'restart']
FOXGLOVE_PORT = 8765
ROSBRIDGE_PORT = 9090


def tcp_clients(port):
    """Established TCP connections to a local port (from /proc, no extra deps)."""
    count = 0
    for table in ('/proc/net/tcp', '/proc/net/tcp6'):
        try:
            with open(table) as f:
                next(f)
                for line in f:
                    local, state = line.split()[1], line.split()[3]
                    if state == '01' and int(local.rsplit(':', 1)[1], 16) == port:
                        count += 1
        except OSError:
            pass
    return count


# Web UI Services menu: id -> (label, what it does, its on/off service).
SERVICE_MENU = {
    'cameras': ('Cameras', 'Each runs only while used. Off: no camera views, stereo or dock search.', '/cameras/set_enabled'),
    'stereo': ('Stereo depth', 'Extra obstacle stops when driving by hand, Depth view. Lidar still protects the robot.', '/depth/set_enabled'),
    'imu': ('IMU (gyro)', 'Turn tracking. Off: odometry falls back to the tracks.', '/services/imu/set_enabled'),
    'sensors': ('Sensor and LED board', 'Front distance, temperature, lights.', '/services/sensors/set_enabled'),
    'crash': ('Crash detection', 'Stops the robot after a hit and marks the spot on the map. Needs the IMU.', '/services/crash/set_enabled'),
    'foxglove': ('Foxglove', 'Remote viewing on port 8765. Off: Foxglove can\'t connect.', '/services/foxglove/set_enabled'),
    'rosbridge': ('rosbridge', 'Web and phone ROS clients on port 9090.', '/services/rosbridge/set_enabled'),
}
COSTMAP_TOPICS = {'global': '/global_costmap/costmap', 'local': '/local_costmap/costmap'}


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wait_future(future, timeout):
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout):
        return None
    return future.result()


class WebUI(Node):
    def __init__(self):
        super().__init__('web_ui')
        self.port = self.declare_parameter('port', 8080).value
        self.drive_yaml = os.path.expanduser(self.declare_parameter(
            'drive_yaml', '~/robot_ws/src/robot_bringup/config/drive.yaml').value)
        self.lock = threading.Lock()
        group = ReentrantCallbackGroup()

        self.tf = Buffer()
        TransformListener(self.tf, self)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.map_msg, self.map_seq, self.map_cache = None, 0, None
        self.create_subscription(OccupancyGrid, '/map', self.on_map, latched)
        self.mode = 'unknown'
        self.create_subscription(String, '/robot/nav_mode', self.on_mode, latched)
        self.odom, self.odom_time = None, 0.0
        # Raw track odometry: calibration must not see the EKF's lidar/gyro corrections.
        self.create_subscription(Odometry, '/wheel/odom', self.on_odom, 10)
        # Per-track encoder view (pico_bridge wheel/joint_states) and direct per-track commands.
        self.ticks_per_rev = self.declare_parameter('encoder_ticks_per_rev', 748.0).value  # = firmware TICKS_PER_REV
        self.wheels, self.wheels_time = None, 0.0
        self.create_subscription(JointState, '/wheel/joint_states', self.on_wheels, 10)
        self.wheel_cmd_pub = self.create_publisher(Float64MultiArray, '/wheel/cmd', 10)

        # /scan on demand (see module docstring).
        self.scan_sub, self.scan, self.scan_time, self.scan_wanted = None, None, 0.0, 0.0

        # Nav2 costmaps on demand, same idea as /scan. layer -> {sub, msg, seq, time}
        self.costmaps = {name: {'sub': None, 'msg': None, 'seq': 0, 'time': 0.0} for name in COSTMAP_TOPICS}
        self.costmap_wanted = 0.0
        # /imu/data on demand, for the /imu page. Rows: seq, time, gyro xyz, accel xyz.
        self.imu_sub, self.imu_wanted, self.imu_seq = None, 0.0, 0
        self.imu_rows = collections.deque(maxlen=1800)  # 36 s at 50 Hz
        self.imu_frame, self.imu_yaw = '', None

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.manual_pub = self.create_publisher(Twist, '/cmd_vel_manual', 10)  # joystick; see drive_guard.py
        self.nav_action = ActionClient(self, NavigateToPose, '/navigate_to_pose', callback_group=group)
        self.goal = {'status': 'none', 'message': '', 'distance': None}
        self.goal_handle = None
        self.map_dir = os.path.expanduser(self.declare_parameter('map_dir', '~/robot_ws/maps').value)
        self.nav_map = ''
        self.create_subscription(String, '/robot/nav_map', lambda m: setattr(self, 'nav_map', m.data), latched)
        self.guard = ''
        self.explore = ''
        self.localize = ''
        self.dock = ''
        # Nav2's planned route (map frame), drawn on the map while a goal is active.
        self.plan, self.plan_time = [], 0.0
        self.create_subscription(Path, '/plan', self.on_plan, 10)
        self.dock_cache = (None, 0.0, None)  # (path, mtime, pose) of maps/<map>.dock.json
        self.create_subscription(String, '/dock/status', lambda m: setattr(self, 'dock', m.data), latched)
        self.create_subscription(String, '/robot/localize_status', lambda m: setattr(self, 'localize', m.data), latched)
        self.battery, self.battery_time, self.battery_status = None, 0.0, ''
        self.create_subscription(BatteryState, '/battery', self.on_battery, 10)
        self.create_subscription(String, '/battery/status', lambda m: setattr(self, 'battery_status', m.data), latched)
        self.charger = None  # /charger/connected from the sensor board's charge-port sense
        self.create_subscription(Bool, '/charger/connected', lambda m: setattr(self, 'charger', m.data), latched)
        self.charger_v = (None, 0.0)  # (volts at the charge port, monotonic time)
        self.create_subscription(Float32, '/charger/voltage',
                                 lambda m: setattr(self, 'charger_v', (m.data, time.monotonic())), 10)
        self.create_subscription(String, '/explore/status', lambda m: setattr(self, 'explore', m.data), latched)
        self.create_subscription(String, '/drive_guard/status', lambda m: setattr(self, 'guard', m.data), latched)
        # Sensor + LED Pico (pico_io.py): lights status, front range, temperature/humidity.
        self.lights, self.lights_time = {}, 0.0
        self.create_subscription(String, '/lights/status', self.on_lights, latched)
        self.create_subscription(Bool, '/depth/enabled', lambda m: self.service_on.__setitem__('stereo', m.data), latched)
        self.crash_marks = []
        self.create_subscription(String, '/crash/list', lambda m: setattr(self, 'crash_marks', json.loads(m.data)), latched)
        self.crash_remove_pub = self.create_publisher(Point, '/crash/remove', 10)
        self.imu_status = ''
        self.create_subscription(String, '/imu/status', lambda m: setattr(self, 'imu_status', m.data), latched)
        self.headlights_pub = self.create_publisher(String, '/lights/headlights', 10)
        self.light_zone_pub = self.create_publisher(String, '/lights/zone', 10)
        self.light_color_pub = self.create_publisher(String, '/lights/color', 10)
        self.io = {}  # name -> (value, monotonic time)
        self.create_subscription(Range, '/range/front', lambda m: self.io_value('range', m.range), 10)
        self.create_subscription(Temperature, '/temperature', lambda m: self.io_value('temp', m.temperature), 10)
        self.create_subscription(RelativeHumidity, '/humidity',
                                 lambda m: self.io_value('humidity', m.relative_humidity * 100.0), 10)
        self.thumbs = {}  # name -> (mtime, png bytes)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)

        self.nav_clients = {c: self.create_client(Trigger, f'/robot/{c}', callback_group=group)
                            for c in NAV_COMMANDS}
        self.get_params = self.create_client(GetParameters, '/pico_bridge/get_parameters',
                                             callback_group=group)
        self.set_params = self.create_client(SetParameters, '/pico_bridge/set_parameters',
                                             callback_group=group)
        self.calib = {}
        self.nav_params = self.create_client(SetParameters, '/nav_manager/set_parameters', callback_group=group)
        self.guard_override = self.create_client(SetBool, '/drive_guard/override', callback_group=group)
        self.service_clients = {name: self.create_client(SetBool, item[2], callback_group=group)
                                for name, item in SERVICE_MENU.items()}
        self.service_on, self.cameras_detail = {}, ''
        self.map_load = self.create_client(LoadMap, '/map_server/load_map', callback_group=group)
        self.mask_load = self.create_client(LoadMap, '/filter_mask_server/load_map', callback_group=group)
        self.create_subscription(String, '/services/status', self.on_services, latched)
        self.create_subscription(String, '/cameras/status', self.on_cameras_status, latched)

        # Manual / calibration motion: published only while fresh, so Nav2 is not fought.
        self.motion_manual = False
        self.motion = (0.0, 0.0)
        self.motion_until = 0.0
        self.test = {'name': None, 'status': 'idle', 'message': '', 'result': None}
        self.test_abort = threading.Event()

        self.lidar_port = self.declare_parameter('lidar_port', '/dev/rplidar').value
        self.lidar_hold = None
        self.create_timer(0.05, self.tick)
        self.create_timer(1.0, self.park_lidar)
        # Camera previews, also on demand: /<side>/preview/image/compressed is JPEG already.
        self.cams = {side: {'sub': None, 'frame': None, 'seq': 0, 'wanted': 0.0,
                            'cond': threading.Condition()} for side in CAMERAS}
        self.create_timer(0.1, self.manage_scan_sub)

        # Stereo calibration (/calibrate): runs only while that page is polling.
        self.stereo = StereoSession(os.path.expanduser('~/robot_ws/calibration_captures'))
        self.calib_dir = os.path.join(os.path.dirname(self.drive_yaml), 'calibration')
        self.urdf_path = os.path.expanduser('~/robot_ws/src/robot_description/urdf/robot.urdf.xacro')
        self.stereo_wanted, self.stereo_auto = 0.0, False
        self.stereo_live = {side: {'found': False, 'corners': []} for side in SIDES}
        self.stereo_steady_since, self.stereo_prev = None, None
        self.stereo_msg, self.stereo_busy, self.stereo_last_capture = '', False, 0.0
        self.stereo_job = {'status': 'idle', 'message': '', 'summary': None}
        self.stereo_lock = threading.Lock()
        # Full-res frames for captures. Subscribed by manage_scan_sub (executor thread) while
        # the calibration page is open, never created/destroyed from HTTP threads: doing that
        # raced the executor (InvalidHandle) and killed it.
        self.raw = {side: {'sub': None, 'frames': [], 'got': 0} for side in SIDES}
        self.raw_wanted = 0.0
        threading.Thread(target=self.stereo_loop, daemon=True).start()

        # Idle auto-stop (see module docstring); 0 disables it. Live: ros2 param set /web_ui idle_timeout <s>
        self.idle_timeout = self.declare_parameter('idle_timeout', 1800.0).value
        self.last_active, self.idle_reason = time.monotonic(), 'starting'
        self.create_timer(5.0, self.check_idle)

    # ---------- ROS callbacks ----------
    def touch(self, reason):
        self.last_active, self.idle_reason = time.monotonic(), reason

    def on_map(self, msg):
        with self.lock:
            self.map_msg, self.map_cache = msg, None
            self.map_seq += 1

    def on_mode(self, msg):
        self.mode = msg.data

    def on_odom(self, msg):
        self.odom, self.odom_time = msg, time.monotonic()

    def on_plan(self, msg):
        pts = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        step = max(1, len(pts) // 200)  # plenty for a line on a web map
        self.plan = [[round(x, 3), round(y, 3)] for x, y in pts[::step] + pts[-1:]] if pts else []
        self.plan_time = time.monotonic()

    def dock_pose(self):
        """Where the robot parks at the dock on the current map (dock.py saves it), or None."""
        if self.mode in ('mapping', 'exploring'):
            path = os.path.join(self.map_dir, '.pending_dock.json')  # seen during this mapping run
        elif self.nav_map:
            path = os.path.join(self.map_dir, self.nav_map + '.dock.json')
        else:
            return None
        if not os.path.exists(path):
            return None
        mtime = os.path.getmtime(path)
        if self.dock_cache[0] != path or self.dock_cache[1] != mtime:
            try:
                with open(path) as f:
                    self.dock_cache = (path, mtime, json.load(f))
            except (OSError, ValueError):
                return None
        return self.dock_cache[2]

    def on_battery(self, msg):
        self.battery, self.battery_time = msg, time.monotonic()

    def battery_payload(self):
        b = self.battery
        if b is None or time.monotonic() - self.battery_time > 3.0 or not b.present:
            return {'present': False, 'status': self.battery_status}
        return {'present': True, 'voltage': round(b.voltage, 2), 'percent': round(b.percentage * 100),
                'status': self.battery_status}

    def on_lights(self, msg):
        try:
            self.lights, self.lights_time = json.loads(msg.data), time.monotonic()
        except ValueError:
            pass

    def io_value(self, name, value):
        self.io[name] = (value, time.monotonic())

    def io_payload(self):
        out = dict(self.lights)
        out['connected'] = bool(out.get('connected')) and time.monotonic() - self.lights_time < 5.0
        for name, max_age in (('range', 1.0), ('temp', 6.0), ('humidity', 6.0)):
            value, t = self.io.get(name, (None, 0.0))
            fresh = value is not None and time.monotonic() - t < max_age
            # JSON has no inf: the range reads 'clear' (null with fresh=True) when nothing is in view.
            out[name] = round(value, 3) if fresh and math.isfinite(value) else None
            out[name + '_ok'] = fresh
        return out

    def set_headlights(self, level):
        if level not in ('off', 'dim', 'on'):
            raise ValueError('headlights: off, dim or on')
        self.headlights_pub.publish(String(data=level))
        return {'ok': True}

    def set_light_zone(self, zone, mode):
        if zone not in ('sides', 'back', 'front') or mode not in ('auto', 'off', 'on'):
            raise ValueError('lights: zone sides, back or front; mode auto, off or on')
        self.light_zone_pub.publish(String(data=f'{zone} {mode}'))
        return {'ok': True}

    def set_light_color(self, zone, color):
        if zone not in ('headlights', 'sides', 'back', 'front') or not isinstance(color, str) \
                or not re.fullmatch(r'#[0-9a-fA-F]{6}', color):
            raise ValueError('lights: zone headlights, sides, back or front; color #rrggbb')
        self.light_color_pub.publish(String(data=f'{zone} {color}'))
        return {'ok': True}

    def on_wheels(self, msg):
        self.wheels, self.wheels_time = msg, time.monotonic()
        # Idle auto-stop: real track motion from any source (web UI, CLI, Nav2, teleop).
        # Not a /cmd_vel subscription: this node publishes /cmd_vel itself, and Fast DDS 2.6
        # can deadlock delivering a node's own messages to itself while peers come and go.
        if any(abs(v) > 0.3 for v in msg.velocity):
            self.touch('driving')

    def on_scan(self, msg):
        self.scan, self.scan_time = msg, time.monotonic()

    def lazy_sub(self, msg_type, topic, callback, qos):
        """Subscription that manage_scan_sub creates and destroys on demand, in a callback
        group of its own: if a message is taken just as it is destroyed, rclpy fails before
        releasing the group, and a shared group would freeze every other callback with it."""
        return self.create_subscription(msg_type, topic, callback, qos,
                                        callback_group=MutuallyExclusiveCallbackGroup())

    def manage_scan_sub(self):
        wanted = time.monotonic() - self.scan_wanted < 8.0
        if wanted and self.scan_sub is None:
            self.scan_sub = self.lazy_sub(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        elif not wanted and self.scan_sub is not None:
            self.destroy_subscription(self.scan_sub)
            self.scan_sub, self.scan = None, None
        imu_wanted = time.monotonic() - self.imu_wanted < 8.0
        if imu_wanted and self.imu_sub is None:
            self.imu_sub = self.lazy_sub(Imu, '/imu/data', self.on_imu, qos_profile_sensor_data)
        elif not imu_wanted and self.imu_sub is not None:
            self.destroy_subscription(self.imu_sub)
            self.imu_sub = None
            self.imu_rows.clear()
        cost_wanted = time.monotonic() - self.costmap_wanted < 8.0
        for name, layer in self.costmaps.items():
            if cost_wanted and layer['sub'] is None:
                layer['sub'] = self.lazy_sub(
                    OccupancyGrid, COSTMAP_TOPICS[name], lambda msg, layer=layer: self.on_costmap(layer, msg),
                    QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                               reliability=ReliabilityPolicy.RELIABLE))
            elif not cost_wanted and layer['sub'] is not None:
                self.destroy_subscription(layer['sub'])
                layer['sub'], layer['msg'] = None, None
        raw_wanted = time.monotonic() - self.raw_wanted < 10.0
        for side, raw in self.raw.items():
            if raw_wanted and raw['sub'] is None:
                raw['sub'] = self.lazy_sub(
                    Image, f'/{side}/image_raw', lambda msg, raw=raw: self.on_raw(raw, msg), qos_profile_sensor_data)
            elif not raw_wanted and raw['sub'] is not None:
                self.destroy_subscription(raw['sub'])
                raw['sub'], raw['frames'] = None, []
        for side, cam in self.cams.items():
            wanted = time.monotonic() - cam['wanted'] < 3.0
            if wanted and cam['sub'] is None:
                cam['sub'] = self.lazy_sub(
                    CompressedImage, CAMERA_TOPICS[side],
                    lambda msg, cam=cam: self.on_frame(cam, msg), qos_profile_sensor_data)
            elif not wanted and cam['sub'] is not None:
                self.destroy_subscription(cam['sub'])
                cam['sub'] = None

    def on_imu(self, msg):
        g, a = msg.angular_velocity, msg.linear_acceleration
        self.imu_seq += 1
        self.imu_frame = msg.header.frame_id
        self.imu_rows.append((self.imu_seq, time.monotonic(), g.x, g.y, g.z, a.x, a.y, a.z))

    def on_costmap(self, layer, msg):
        layer['msg'], layer['seq'], layer['time'] = msg, layer['seq'] + 1, time.monotonic()

    def on_raw(self, raw, msg):
        raw['got'] += 1
        raw['frames'] = (raw['frames'] + [msg])[-10:]

    def on_frame(self, cam, msg):
        with cam['cond']:
            cam['frame'] = bytes(msg.data)
            cam['seq'] += 1
            cam['cond'].notify_all()

    def stream(self, handler, side):
        """multipart/x-mixed-replace MJPEG: an <img> shows it with no JavaScript."""
        cam = self.cams[side]
        handler.send_response(200)
        handler.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        handler.send_header('Cache-Control', 'no-store')
        handler.end_headers()
        seq, last_frame = cam['seq'], time.monotonic()
        while time.monotonic() - last_frame < 40.0:  # ends if the camera stops; the page retries. Starting takes ~20 s
            cam['wanted'] = time.monotonic()
            with cam['cond']:
                cam['cond'].wait_for(lambda: cam['seq'] != seq, timeout=1.0)
                if cam['seq'] == seq:
                    continue
                seq, frame = cam['seq'], cam['frame']
            last_frame = time.monotonic()
            try:
                handler.wfile.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: '
                                    + str(len(frame)).encode() + b'\r\n\r\n' + frame + b'\r\n')
            except (BrokenPipeError, ConnectionResetError):
                return

    def tick(self):
        if time.monotonic() < self.motion_until:
            t = Twist()
            t.linear.x, t.angular.z = self.motion
            (self.manual_pub if self.motion_manual else self.cmd_pub).publish(t)

    def on_services(self, msg):
        self.service_on.update(json.loads(msg.data))

    def on_cameras_status(self, msg):
        self.service_on['cameras'] = msg.data != 'disabled'
        self.cameras_detail = msg.data

    def services_payload(self):
        return [{'id': name, 'label': item[0], 'help': item[1],
                 'on': self.service_on.get(name) if self.service_clients[name].service_is_ready() else None}
                for name, item in SERVICE_MENU.items()]

    def set_service(self, name, on):
        client = self.service_clients[name]
        if not client.wait_for_service(timeout_sec=1.0):
            raise RuntimeError(f'{SERVICE_MENU[name][0]} can\'t be changed: is the stack running? Restart it if so')
        if wait_future(client.call_async(SetBool.Request(data=on)), 15.0) is None:
            raise RuntimeError(f'{SERVICE_MENU[name][0]} didn\'t answer')
        self.service_on[name] = on

    def set_guard_override(self, on):
        if not self.guard_override.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('drive guard is not responding: restart the stack')
        res = wait_future(self.guard_override.call_async(SetBool.Request(data=on)), 3.0)
        if res is None:
            raise RuntimeError('drive guard is not responding: restart the stack')
        self.guard = res.message

    def drive(self, v, w, hold=0.3, manual=False):
        self.motion_manual = manual
        self.motion = (max(-MAX_V, min(MAX_V, v)), max(-MAX_W, min(MAX_W, w)))
        self.motion_until = time.monotonic() + hold

    def halt(self):
        self.motion_manual = False
        self.motion = (0.0, 0.0)
        self.motion_until = time.monotonic() + 0.3  # a few explicit zeros

    # ---------- frames ----------
    def world_frame(self):
        return 'map' if self.tf.can_transform('map', 'base_link', Time()) else 'odom'

    def pose_in(self, world, child):
        try:
            t = self.tf.lookup_transform(world, child, Time()).transform
        except Exception:
            return None
        return {'x': t.translation.x, 'y': t.translation.y, 'yaw': yaw_of(t.rotation)}

    # ---------- pico_bridge parameters ----------
    def read_calib(self):
        if not self.get_params.wait_for_service(timeout_sec=1.0):
            return None
        res = wait_future(self.get_params.call_async(GetParameters.Request(names=CALIB_PARAMS)), 3.0)
        if res is None:
            return None
        out = {}
        for name, v in zip(CALIB_PARAMS, res.values):
            out[name] = v.double_value if v.type == ParameterType.PARAMETER_DOUBLE else v.bool_value
        self.calib = out
        return out

    def apply_calib(self, values):
        params = []
        for name, value in values.items():
            if name not in CALIB_PARAMS:
                raise ValueError(f'unknown parameter {name}')
            if name in ('wheel_radius', 'track_width'):
                value = float(value)
                if not 0.005 < value < 1.0:
                    raise ValueError(f'{name}={value} out of range')
                pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=value)
            else:
                pv = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=bool(value))
            params.append(Parameter(name=name, value=pv))
        if not self.set_params.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('pico_bridge is not running')
        res = wait_future(self.set_params.call_async(SetParameters.Request(parameters=params)), 3.0)
        if res is None or not all(r.successful for r in res.results):
            raise RuntimeError('pico_bridge rejected: ' + (res and ', '.join(r.reason for r in res.results) or 'timeout'))
        self.persist(values)
        return self.read_calib()

    def persist(self, values):
        """Rewrite just the value on each line of drive.yaml, keeping comments."""
        with open(self.drive_yaml) as f:
            text = f.read()
        for name, value in values.items():
            literal = ('true' if value else 'false') if isinstance(value, bool) else f'{float(value):.4f}'
            pattern = re.compile(rf'^(\s*{name}:\s*)(\S+)', re.M)
            if pattern.search(text):
                text = pattern.sub(lambda m: m.group(1) + literal, text, count=1)
            else:
                text = text.rstrip('\n') + f'\n    {name}: {literal}\n'
        with open(self.drive_yaml, 'w') as f:
            f.write(text)

    # ---------- calibration tests ----------
    def odom_state(self):
        o = self.odom
        if o is None:
            return None
        p = o.pose.pose
        return p.position.x, p.position.y, yaw_of(p.orientation)

    def start_test(self, name):
        if self.test['status'] == 'running':
            raise RuntimeError('a test is already running')
        if self.mode in ('mapping', 'navigating', 'exploring'):
            raise RuntimeError('stop mapping/navigation first, Nav2 would fight the test')
        if time.monotonic() - self.odom_time > 1.0:
            raise RuntimeError('no /wheel/odom from pico_bridge: is the Pico connected?')
        calib = self.read_calib() or self.calib
        if not calib:
            raise RuntimeError('cannot read pico_bridge parameters')
        self.test_abort.clear()
        self.test = {'name': name, 'status': 'running', 'message': 'starting', 'result': None}
        threading.Thread(target=self.run_test, args=(name, calib), daemon=True).start()

    def run_test(self, name, calib):
        try:
            if name in ('left', 'right'):
                # One track forward at ~4 rad/s sprocket speed, the other still.
                s = 4.0 * calib['wheel_radius']
                v = s / 2.0
                w = (s if name == 'right' else -s) / calib['track_width']
                self.test['message'] = f'{name} track forward for 2 s: watch which track moves and which way'
                self.timed(v, w, 2.0)
                result = None
            elif name == 'straight':
                result = self.until(0.15, 0.0, lambda d, a: d >= 1.0, 15.0,
                                    'driving 1 m by odometry')
            elif name == 'spin':
                result = self.until(0.0, 0.8, lambda d, a: abs(a) >= 2 * math.pi, 30.0,
                                    'spinning 360 degrees by odometry (counter-clockwise)')
            else:
                raise ValueError(name)
            self.test.update(status='aborted' if self.test_abort.is_set() else 'done',
                             message='aborted' if self.test_abort.is_set() else 'done', result=result)
        except Exception as e:  # report to the page instead of dying silently
            self.halt()
            self.test.update(status='error', message=str(e))

    def timed(self, v, w, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not self.test_abort.is_set():
            self.drive(v, w)
            time.sleep(0.05)
        self.halt()

    def until(self, v, w, done, timeout, message):
        start = self.odom_state()
        dist, ang, last_yaw = 0.0, 0.0, start[2]
        end = time.monotonic() + timeout
        self.test['message'] = message
        while not self.test_abort.is_set():
            x, y, yaw = self.odom_state()
            dist = math.hypot(x - start[0], y - start[1])
            ang += math.atan2(math.sin(yaw - last_yaw), math.cos(yaw - last_yaw))
            last_yaw = yaw
            if done(dist, ang):
                break
            if time.monotonic() > end:
                self.halt()
                raise RuntimeError('timed out: the robot did not reach the target by odometry')
            self.drive(v, w)
            self.test['message'] = f'{message}: {dist:.2f} m, {math.degrees(ang):.0f} deg'
            time.sleep(0.05)
        self.halt()
        time.sleep(0.6)  # let it coast to a stop, then take the final reading
        x, y, yaw = self.odom_state()
        ang += math.atan2(math.sin(yaw - last_yaw), math.cos(yaw - last_yaw))
        return {'odom_distance': math.hypot(x - start[0], y - start[1]),
                'odom_degrees': math.degrees(ang)}

    # ---------- robot stack (systemd) ----------
    def stack_state(self):
        r = subprocess.run(['systemctl', 'is-active', STACK_UNIT], capture_output=True, text=True)
        return r.stdout.strip() or 'unknown'

    def park_lidar(self):
        down = self.stack_state() in ('inactive', 'failed')
        if down and self.lidar_hold is None and os.path.exists(self.lidar_port):
            try:
                # pyserial asserts DTR on open, which is "motor off" on the A1.
                self.lidar_hold = serial.Serial(self.lidar_port, 115200)
                self.lidar_hold.dtr = True
                self.get_logger().info('stack down: holding lidar port, motor off')
            except serial.SerialException as e:
                self.get_logger().warn(f'cannot hold {self.lidar_port}: {e}', throttle_duration_sec=30)
        elif not down:
            self.release_lidar()

    def release_lidar(self):
        if self.lidar_hold is not None:
            try:
                self.lidar_hold.close()
            except Exception:
                pass
            self.lidar_hold = None
            self.get_logger().info('released lidar port for the stack')

    def check_idle(self):
        self.idle_timeout = float(self.get_parameter('idle_timeout').value)
        if self.mode in ('mapping', 'navigating', 'exploring'):
            self.touch(self.mode)
        elif tcp_clients(FOXGLOVE_PORT):
            self.touch('Foxglove connected')
        elif tcp_clients(ROSBRIDGE_PORT):
            self.touch('rosbridge client connected')
        if self.stack_state() != 'active':
            self.touch('stack not running')  # a fresh start gets the full timeout
            return
        if self.idle_timeout > 0 and time.monotonic() - self.last_active > self.idle_timeout:
            self.get_logger().warn(f'idle for {self.idle_timeout / 60:.0f} min: stopping the stack')
            self.stack('stop')
            self.touch('stopped after idle timeout')

    def stack(self, action):
        if action in ('stop', 'restart'):
            self.test_abort.set()
            self.halt()
            time.sleep(0.3)  # let the zero commands reach the Pico before the bridge goes
        if action in ('start', 'restart'):
            self.release_lidar()
        # --no-block: stopping takes up to 15 s; the page follows it through /api/state.
        r = subprocess.run(['sudo', '-n', 'systemctl', '--no-block', action, STACK_UNIT],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            return {'success': False, 'message': (r.stderr or r.stdout).strip()[-300:]}
        return {'success': True, 'message': f'stack {action} requested'}

    # ---------- stereo calibration ----------
    def stereo_loop(self):
        """Board detection on the previews, steadiness, and auto-capture."""
        while rclpy.ok():
            time.sleep(0.25)
            if time.monotonic() - self.stereo_wanted > 3.0:
                self.stereo_steady_since = self.stereo_prev = None
                continue
            found = {}
            for side in SIDES:
                cam = self.cams[side]
                cam['wanted'] = time.monotonic()  # keep the preview subscribed
                frame = cam['frame']
                c = None
                if frame:
                    img = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        c = find_corners(img, self.stereo.pattern, accurate=False)
                        if c is not None:
                            found[side] = (c, (img.shape[1], img.shape[0]))
                self.stereo_live[side] = {
                    'found': c is not None,
                    'corners': [] if c is None else
                    [[round(float(x) / img.shape[1], 4), round(float(y) / img.shape[0], 4)] for x, y in c]}
            if len(found) < 2:
                self.stereo_steady_since = self.stereo_prev = None
                continue
            cur = {side: found[side][0] for side in SIDES}
            moved = max(float(np.abs(cur[s] - self.stereo_prev[s]).mean()) for s in SIDES) if self.stereo_prev else 99
            self.stereo_prev = cur
            now = time.monotonic()
            if moved > 0.8:  # preview pixels (640 wide)
                self.stereo_steady_since = None
            elif self.stereo_steady_since is None:
                self.stereo_steady_since = now
            c, size = found['left']
            new_pose = self.stereo.novelty(sample_params(c, self.stereo.cols, size)) >= NOVELTY
            steady = self.stereo_steady_since is not None and now - self.stereo_steady_since > 1.0
            if (self.stereo_auto and steady and new_pose and not self.stereo_busy
                    and now - self.stereo_last_capture > 1.5):
                self.stereo_capture()

    def grab_pair(self, timeout=3.0, next_left=False):
        """Newest full-res left/right frames taken within 20 ms of each other; with
        next_left, also a left frame >= 0.08 s later (for the board-moved check)."""
        for raw in self.raw.values():
            raw['frames'], raw['got'] = [], 0  # only frames from now on
        end = time.monotonic() + timeout + 1.0  # + time to subscribe if the page just opened
        while time.monotonic() < end:
            self.raw_wanted = time.monotonic()
            time.sleep(0.05)
            frames = {side: list(self.raw[side]['frames']) for side in SIDES}
            best = None
            for a in frames['left']:
                for b in frames['right']:
                    dt = abs((Time.from_msg(a.header.stamp) - Time.from_msg(b.header.stamp)).nanoseconds) * 1e-9
                    if dt < 0.02 and (best is None or dt < best[0]):
                        best = (dt, a, b)
            if best and not next_left:
                return {'left': best[1], 'right': best[2]}
            if best:
                t0 = Time.from_msg(best[1].header.stamp).nanoseconds
                later = [m for m in frames['left'] if Time.from_msg(m.header.stamp).nanoseconds - t0 >= 8e7]
                if later:
                    return {'left': best[1], 'right': best[2], 'left_next': later[0]}
        raise RuntimeError(f'no matching left/right camera frames in {timeout:.0f} s '
                           f'(got {self.raw["left"]["got"]} left, {self.raw["right"]["got"]} right: '
                           'is the stack running?)')

    @staticmethod
    def image_arrays(msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
        ch = {'rgb8': 3, 'bgr8': 3, 'mono8': 1, 'rgba8': 4, 'bgra8': 4}.get(msg.encoding)
        if ch is None:
            raise RuntimeError(f'unsupported camera encoding {msg.encoding}')
        img = img[:, :msg.width * ch].reshape(msg.height, msg.width, ch) if ch > 1 else img[:, :msg.width]
        from robot_bringup.stereo_calib import to_gray
        gray = to_gray(img, msg.encoding)
        color = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) if msg.encoding == 'rgb8' else img
        return gray, color

    def stereo_capture(self):
        if self.stereo_busy:
            raise RuntimeError('busy: wait for the current step to finish')
        self.stereo_busy = True
        self.stereo_last_capture = time.monotonic()
        try:
            self.stereo_msg = 'capturing…'
            pair = self.grab_pair(next_left=True)
            arrays = {side: self.image_arrays(pair[side]) for side in SIDES}
            later = self.image_arrays(pair['left_next'])[0]
            with self.stereo_lock:
                self.stereo_msg = self.stereo.add({s: arrays[s][0] for s in SIDES},
                                                  {s: arrays[s][1] for s in SIDES}, left_next=later)
            self.stereo_job = {'status': 'idle', 'message': '', 'summary': None}
        except Exception as e:
            self.stereo_msg = str(e)
            raise
        finally:
            self.stereo_busy = False
            self.stereo_last_capture = time.monotonic()
        return self.stereo_msg

    def stereo_compute(self):
        if self.stereo_busy:
            raise RuntimeError('busy: wait for the current step to finish')
        def run():
            self.stereo_busy = True
            self.stereo_job = {'status': 'running', 'message': 'calibrating (takes a few seconds)…', 'summary': None}
            try:
                with self.stereo_lock:
                    summary = self.stereo.compute()
                self.stereo_job = {'status': 'done', 'message': 'calibrated', 'summary': summary}
            except Exception as e:
                self.stereo_job = {'status': 'error', 'message': str(e), 'summary': None}
            finally:
                self.stereo_busy = False
        threading.Thread(target=run, daemon=True).start()

    def stereo_rectified(self, fresh):
        pair = None
        if fresh:
            raw = self.grab_pair()
            pair = {side: self.image_arrays(raw[side])[0] for side in SIDES}
        return self.stereo.rectified_jpeg(pair)

    def stereo_save(self):
        with self.stereo_lock:
            backup = self.stereo.write_yaml(self.calib_dir, self.urdf_path)
        r = self.stack('restart')  # camera nodes and the URDF baseline load at startup
        return {'success': True, 'message': f'saved (old files in {os.path.basename(backup)}); '
                                            + ('restarting the robot stack' if r['success'] else r['message'])}

    def stereo_state(self):
        self.stereo_wanted = time.monotonic()  # raw frames are only subscribed around captures (grab_pair)
        st = self.stereo
        steady = self.stereo_steady_since is not None and time.monotonic() - self.stereo_steady_since > 1.0
        return {
            'pattern': {'cols': st.cols, 'rows': st.rows, 'square_mm': st.square_mm},
            'live': self.stereo_live, 'steady': steady, 'auto': self.stereo_auto,
            'busy': self.stereo_busy, 'message': self.stereo_msg,
            'samples': len(st.samples), 'coverage': st.progress(), 'job': self.stereo_job,
            'stack': self.stack_state(),
        }

    # ---------- API payloads ----------
    def state(self):
        now = time.monotonic()
        world = self.world_frame()
        return {
            'stack': self.stack_state(),
            'mode': self.mode if self.nav_clients['stop'].service_is_ready() else 'off',
            'nav_map': self.nav_map,
            'guard': self.guard,
            'explore': self.explore,
            'localize': self.localize,
            'dock': self.dock,
            'dock_pose': self.dock_pose(),
            'plan': self.plan if time.monotonic() - self.plan_time < 3.0 and world == 'map' else [],
            'battery': self.battery_payload(),
            'charger': self.charger,
            'charger_voltage': (round(self.charger_v[0], 2) if self.charger_v[0] is not None
                                and time.monotonic() - self.charger_v[1] < 2.0 else None),
            'io': self.io_payload(),
            # Latched, so it outlives the driver: only believed while the driver is there.
            'imu': self.imu_status if self.count_publishers('/imu/data') else '',
            'services': self.services_payload(),
            'crash_marks': self.crash_marks,
            'localized': world == 'map',
            'goal': self.goal,
            'frame': world,
            'pose': self.pose_in(world, 'base_link'),
            'pico': now - self.odom_time < 1.0,
            'lidar': self.scan_sub is not None and now - self.scan_time < 1.0,
            'map_seq': self.map_seq if self.map_msg else 0,
            'calib': self.calib,
            'test': self.test,
            'idle': {'timeout': self.idle_timeout, 'reason': self.idle_reason,
                     'stop_in': max(0.0, self.idle_timeout - (now - self.last_active))},
        }

    def map_payload(self):
        with self.lock:
            if self.map_msg is None:
                return None
            if self.map_cache is None:
                m = self.map_msg
                o = m.info.origin
                self.map_cache = json.dumps({
                    'seq': self.map_seq,
                    'width': m.info.width, 'height': m.info.height,
                    'resolution': m.info.resolution,
                    'origin': {'x': o.position.x, 'y': o.position.y, 'yaw': yaw_of(o.orientation)},
                    # int8 cells (-1 unknown, 0..100) as raw bytes
                    'data': base64.b64encode(bytes(v & 0xFF for v in m.data)).decode(),
                }).encode()
            return self.map_cache

    def scan_payload(self):
        self.scan_wanted = time.monotonic()
        s = self.scan
        if s is None or time.monotonic() - self.scan_time > 1.0:
            return {'ok': False}
        world = self.world_frame()
        return {
            'ok': True, 'frame': world,
            'laser': self.pose_in(world, s.header.frame_id),
            'angle_min': s.angle_min, 'angle_increment': s.angle_increment,
            'ranges': [round(r, 3) if s.range_min <= r <= min(s.range_max, SCAN_MAX_RANGE) else None for r in s.ranges],
        }

    def costmap_payload(self, have):
        """Costmap layers in the display frame. Cells only for layers newer than the page's copy
        (have: layer -> seq), since the global one is large; the origin always, as odom drifts."""
        self.costmap_wanted = time.monotonic()
        world, layers = self.world_frame(), {}
        for name, layer in self.costmaps.items():
            m = layer['msg']
            if m is None or time.monotonic() - layer['time'] > 5.0:
                continue
            at = self.pose_in(world, m.header.frame_id)
            if at is None:
                continue
            o = m.info.origin
            c, s = math.cos(at['yaw']), math.sin(at['yaw'])
            out = {'seq': layer['seq'], 'width': m.info.width, 'height': m.info.height,
                   'resolution': m.info.resolution,
                   'origin': {'x': at['x'] + c * o.position.x - s * o.position.y,
                              'y': at['y'] + s * o.position.x + c * o.position.y,
                              'yaw': at['yaw'] + yaw_of(o.orientation)}}
            if have.get(name) != layer['seq']:
                # int8 cells (-1 unknown, 0 free, 1-98 inflation, 99 inscribed, 100 lethal) as raw bytes
                out['data'] = base64.b64encode(bytes(m.data)).decode()
            layers[name] = out
        return {'frame': world, 'layers': layers}

    def imu_payload(self, since):
        """Samples newer than since: [time, gyro xyz in deg/s, accel xyz in m/s^2], in the robot's axes
        once the board's mount is known from TF."""
        self.imu_wanted = now = time.monotonic()
        if self.imu_yaw is None and self.imu_frame:
            at = self.pose_in('base_link', self.imu_frame)
            self.imu_yaw = at['yaw'] if at else None
        c, s = (math.cos(self.imu_yaw), math.sin(self.imu_yaw)) if self.imu_yaw is not None else (1.0, 0.0)
        rows, last = [], since
        for seq, t, gx, gy, gz, ax, ay, az in list(self.imu_rows):
            if seq <= since:
                continue
            last = seq
            d = 180.0 / math.pi
            rows.append([round(t, 3), round((c * gx - s * gy) * d, 2), round((s * gx + c * gy) * d, 2), round(gz * d, 2),
                         round(c * ax - s * ay, 3), round(s * ax + c * ay, 3), round(az, 3)])
        fresh = bool(self.imu_rows) and now - self.imu_rows[-1][1] < 2.0
        return {'ok': fresh, 'now': round(now, 3), 'seq': last, 'samples': rows,
                'frame': 'robot' if self.imu_yaw is not None else 'board'}

    def wheels_payload(self):
        w = self.wheels
        if w is None or time.monotonic() - self.wheels_time > 1.0:
            return {'live': False, 'ticks_per_rev': self.ticks_per_rev}
        side = lambda i: {'rad': w.position[i], 'ticks': round(w.position[i] * self.ticks_per_rev / (2 * math.pi)),
                          'speed': w.velocity[i], 'target': w.effort[i]}
        return {'live': True, 'ticks_per_rev': self.ticks_per_rev, 'left': side(0), 'right': side(1)}

    def drive_wheels(self, left, right):
        if self.mode in ('mapping', 'navigating', 'exploring'):
            raise RuntimeError('stop mapping/navigation first, Nav2 would fight the test')
        self.test_abort.set()
        self.motion_until = 0.0  # drop any hold-to-drive /cmd_vel still in flight
        limit = 13.0  # = firmware MAX_RAD_S
        msg = Float64MultiArray()
        msg.data = [max(-limit, min(limit, float(left))), max(-limit, min(limit, float(right)))]
        self.wheel_cmd_pub.publish(msg)

    def call_nav(self, command):
        client = self.nav_clients[command]
        if not client.wait_for_service(timeout_sec=1.0):
            if self.nav_clients['stop'].service_is_ready():  # nav_manager is up, just not this service
                return {'success': False, 'message': f'"{command}" is not available yet: restart the stack '
                                                      '(Robot tab) to load the latest software'}
            return {'success': False, 'message': 'nav_manager is not running (is the stack up?)'}
        self.halt()
        res = wait_future(client.call_async(Trigger.Request()), 45.0)
        if res is None:
            return {'success': False, 'message': 'timed out'}
        return {'success': res.success, 'message': res.message}

    def send_goal(self, x, y, yaw):
        if self.mode == 'exploring':
            raise RuntimeError('exploring on its own: stop it first to send your own goals')
        if self.mode not in ('mapping', 'navigating'):
            raise RuntimeError('start mapping or pick a saved map to navigate on first')
        if self.world_frame() != 'map':
            raise RuntimeError("navigation doesn't know where the robot is yet: use Set pose first"
                               if self.mode == 'navigating' else 'mapping is still starting, try again in a moment')
        if not self.nav_action.wait_for_server(timeout_sec=1.0):
            raise RuntimeError('navigation is still starting, try again in a moment')
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        self.goal = {'status': 'sending', 'message': 'sending goal', 'distance': None}
        handle = wait_future(self.nav_action.send_goal_async(goal, feedback_callback=self.on_goal_feedback), 5.0)
        if handle is None or not handle.accepted:
            self.goal = {'status': 'rejected', 'message': 'navigation refused the goal (still starting?)', 'distance': None}
            raise RuntimeError(self.goal['message'])
        self.goal_handle = handle
        self.goal = {'status': 'active', 'message': 'driving to the goal', 'distance': None}
        handle.get_result_async().add_done_callback(lambda f, h=handle: self.on_goal_result(h, f))

    def on_goal_feedback(self, fb):
        if self.goal['status'] == 'active':
            self.goal['distance'] = round(fb.feedback.distance_remaining, 2)

    def on_goal_result(self, handle, future):
        if handle is not self.goal_handle:
            return  # superseded by a newer goal
        status = future.result().status
        self.goal = {
            GoalStatus.STATUS_SUCCEEDED: {'status': 'succeeded', 'message': 'goal reached'},
            GoalStatus.STATUS_CANCELED: {'status': 'canceled', 'message': 'goal cancelled'},
        }.get(status, {'status': 'failed', 'message': "couldn't reach the goal (blocked, or no path)"})
        self.goal['distance'] = None
        self.goal_handle = None

    def cancel_goal(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()

    # ---------- saved maps ----------
    def list_maps(self):
        maps = []
        for path in glob.glob(os.path.join(self.map_dir, '*.yaml')):
            name = os.path.splitext(os.path.basename(path))[0]
            if name == 'latest' or not MAP_NAME_RE.match(name):
                continue
            info = {'name': name, 'modified': os.path.getmtime(path),
                    'can_continue': os.path.exists(os.path.join(self.map_dir, name + '.posegraph')),
                    'has_pose': os.path.exists(os.path.join(self.map_dir, name + '.pose.json'))}
            try:
                with open(path) as f:
                    res = float(next(l for l in f if l.startswith('resolution:')).split(':')[1])
                from PIL import Image
                with Image.open(os.path.join(self.map_dir, name + '.pgm')) as im:
                    info['size_m'] = [round(im.width * res, 1), round(im.height * res, 1)]
            except Exception:
                pass
            maps.append(info)
        latest = ''
        try:
            with open(os.path.join(self.map_dir, 'latest.yaml')) as f:
                img = next(l for l in f if l.startswith('image:')).split(':', 1)[1].strip()
                latest = os.path.splitext(os.path.basename(img))[0]
        except Exception:
            pass
        for m in maps:
            m['latest'] = m['name'] == latest
        return sorted(maps, key=lambda m: -m['modified'])

    def map_thumb(self, name):
        if not MAP_NAME_RE.match(name):
            return None
        pgm = os.path.join(self.map_dir, name + '.pgm')
        if not os.path.exists(pgm):
            return None
        mtime = os.path.getmtime(pgm)
        cached = self.thumbs.get(name)
        if cached and cached[0] == mtime:
            return cached[1]
        from PIL import Image
        with Image.open(pgm) as im:
            im = im.convert('L')
            box = im.point(lambda v: 255 if v != 205 else 0).getbbox()  # crop the unknown border
            if box:
                im = im.crop(box)
            im.thumbnail((240, 240))
            buf = io.BytesIO()
            im.save(buf, 'PNG')
        self.thumbs[name] = (mtime, buf.getvalue())
        return buf.getvalue()

    def set_nav_param(self, name, value):
        if not self.nav_params.wait_for_service(timeout_sec=1.0):
            raise RuntimeError('nav_manager is not running (is the stack up?)')
        req = SetParameters.Request()
        req.parameters = [Parameter(name=name, value=ParameterValue(type=ParameterType.PARAMETER_STRING,
                                                                   string_value=value))]
        res = wait_future(self.nav_params.call_async(req), 5.0)
        if res is None or not res.results[0].successful:
            raise RuntimeError(f'could not set {name} on nav_manager')

    def save_map(self, name):
        name = name.strip()
        if name and not MAP_NAME_RE.match(name):
            raise RuntimeError('map names: up to 40 letters, digits, - or _ (no spaces)')
        self.set_nav_param('map_name', name)
        return self.call_nav('save_map')

    def navigate_map(self, name):
        if not MAP_NAME_RE.match(name):
            raise RuntimeError('unknown map')
        self.set_nav_param('map_file', name)
        return self.call_nav('start_navigation')

    # ---------- map editor (/mapedit) ----------
    PGM = {'wall': 0, 'unknown': 205, 'free': 254}  # map_saver trinary values

    def map_paths(self, name):
        if not MAP_NAME_RE.match(name) or name == 'latest':
            raise RuntimeError('unknown map')
        base = os.path.join(self.map_dir, name)
        if not os.path.exists(base + '.pgm'):
            raise RuntimeError('no such map')
        return base + '.pgm', base + '.yaml', base + '.orig.pgm'

    def map_edit_info(self, name):
        pgm, yml, orig = self.map_paths(name)
        from PIL import Image
        with Image.open(pgm) as im:
            w, h = im.size
        info = {'name': name, 'width': w, 'height': h, 'has_original': os.path.exists(orig),
                'has_keepout': os.path.exists(self.keepout_paths(name)[1]),
                'in_use': self.mode == 'navigating' and self.nav_map == name}
        info.update(self.map_yaml_geometry(yml))
        return info

    @staticmethod
    def map_yaml_geometry(yml):
        with open(yml) as f:
            doc = yaml.safe_load(f)
        return {'resolution': float(doc['resolution']), 'origin': [float(v) for v in doc['origin'][:2]]}

    def keepout_paths(self, name):
        base = os.path.join(self.map_dir, name)
        return base + '.keepout.pgm', base + '.keepout.yaml'

    def map_edit_keepout_png(self, name):
        """No-go mask on the map's own grid (white = no-go). The mask keeps the origin it was
        saved with, so it stays put in the world when "Continue mapping" grows the map."""
        pgm, yml, _ = self.map_paths(name)
        kpgm, kyml = self.keepout_paths(name)
        from PIL import Image
        with Image.open(pgm) as im:
            w, h = im.size
        out = np.zeros((h, w), np.uint8)
        if os.path.exists(kyml) and os.path.exists(kpgm):
            g, kg = self.map_yaml_geometry(yml), self.map_yaml_geometry(kyml)
            res = g['resolution']
            with Image.open(kpgm) as km:
                mask = np.array(km.convert('L')) < 128
            kh, kw = mask.shape
            # Image row 0 is the top (largest y) of each grid.
            col = int(round((kg['origin'][0] - g['origin'][0]) / res))
            row = int(round((g['origin'][1] + h * res - kg['origin'][1] - kh * res) / res))
            r0, c0 = max(row, 0), max(col, 0)
            r1, c1 = min(row + kh, h), min(col + kw, w)
            if r1 > r0 and c1 > c0:
                out[r0:r1, c0:c1][mask[r0 - row:r1 - row, c0 - col:c1 - col]] = 255
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, 'PNG')
        return buf.getvalue()

    def apply_map_live(self, name):
        """If navigation is using this map, load the saved edits into it now. Returns a note for the user."""
        if self.nav_map != name or self.mode != 'navigating':
            return ''
        def load(client, yml):
            if not client.wait_for_service(timeout_sec=1.0):
                return False
            res = wait_future(client.call_async(LoadMap.Request(map_url=yml)), 10.0)
            return res is not None and res.result == LoadMap.Response.RESULT_SUCCESS
        if not load(self.map_load, self.map_paths(name)[1]):
            return " Navigation couldn't load the edits: restart it."
        kyml = self.keepout_paths(name)[1]
        if not os.path.exists(kyml):
            return ' Navigation updated.'
        if not self.mask_load.service_is_ready():  # navigation started before this map had any zones
            return ' Map updated in navigation. Restart it to apply no-go zones, this is the first on this map.'
        if not load(self.mask_load, kyml):
            return " Map updated in navigation, but its no-go zones couldn't load: restart it."
        return ' Navigation updated.'

    def map_edit_save_keepout(self, name, png_b64):
        _, yml, _ = self.map_paths(name)
        kpgm, kyml = self.keepout_paths(name)
        from PIL import Image
        with Image.open(io.BytesIO(base64.b64decode(png_b64))) as im:
            nogo = np.array(im.convert('L')) > 127
        # Running navigation can't drop its zones, so it gets an empty mask to load instead.
        if not nogo.any() and not (self.nav_map == name and self.mode == 'navigating' and self.mask_load.service_is_ready()):
            for path in (kpgm, kyml):
                if os.path.exists(path):
                    os.remove(path)
            return 0
        # map_server trinary: black = occupied (100) = keepout, white = free.
        Image.fromarray(np.where(nogo, 0, 254).astype(np.uint8)).save(kpgm)
        g = self.map_yaml_geometry(yml)
        with open(kyml, 'w') as f:
            f.write(f'image: {os.path.basename(kpgm)}\nmode: trinary\nresolution: {g["resolution"]}\n'
                    f'origin: [{g["origin"][0]}, {g["origin"][1]}, 0]\nnegate: 0\n'
                    'occupied_thresh: 0.65\nfree_thresh: 0.25\n')
        return int(nogo.sum())

    def map_edit_png(self, name):
        pgm = self.map_paths(name)[0]
        from PIL import Image
        with Image.open(pgm) as im:
            buf = io.BytesIO()
            im.convert('L').save(buf, 'PNG')
        return buf.getvalue()

    def map_edit_save(self, name, png_b64, keepout_b64=None):
        pgm, _, orig = self.map_paths(name)
        from PIL import Image
        import numpy as np_
        with Image.open(io.BytesIO(base64.b64decode(png_b64))) as im:
            a = np_.array(im.convert('L'))
        with Image.open(pgm) as old:
            if a.shape != (old.size[1], old.size[0]):
                raise RuntimeError('edited image has the wrong size')
        # Snap to the three values map_server reads (browser canvases can blend edges).
        out = np_.full(a.shape, self.PGM['unknown'], np_.uint8)
        out[a < 100] = self.PGM['wall']
        out[a > 230] = self.PGM['free']
        if not os.path.exists(orig):
            import shutil
            shutil.copy2(pgm, orig)  # first edit: keep what SLAM made
        Image.fromarray(out).save(pgm)
        self.thumbs.pop(name, None)
        zones = ''
        if keepout_b64 is not None:
            cells = self.map_edit_save_keepout(name, keepout_b64)
            res = self.map_yaml_geometry(self.map_paths(name)[1])['resolution']
            zones = f' No-go zones: {cells * res * res:.1f} m².' if cells else ' No no-go zones.'
        note = self.apply_map_live(name)
        return {'success': True, 'message': f'saved "{name}" (original kept).{zones}{note}'}

    def map_edit_revert(self, name):
        pgm, _, orig = self.map_paths(name)
        if not os.path.exists(orig):
            raise RuntimeError('this map has no saved original')
        os.replace(orig, pgm)
        self.thumbs.pop(name, None)
        return {'success': True, 'message': f'"{name}" is back to the original SLAM map.{self.apply_map_live(name)}'}

    def continue_map(self, name):
        if not MAP_NAME_RE.match(name):
            raise RuntimeError('unknown map')
        self.set_nav_param('map_file', name)
        return self.call_nav('continue_mapping')

    def delete_map(self, name):
        if not MAP_NAME_RE.match(name) or name == 'latest':
            raise RuntimeError('unknown map')
        if self.mode == 'navigating' and self.nav_map == name:
            raise RuntimeError('stop navigating on this map first')
        removed = 0
        for ext in ('.yaml', '.pgm', '.pose.json', '.orig.pgm', '.dock.json', '.posegraph', '.data',
                    '.keepout.pgm', '.keepout.yaml'):
            path = os.path.join(self.map_dir, name + ext)
            if os.path.exists(path):
                os.remove(path)
                removed += 1
        if not removed:
            raise RuntimeError('no such map')
        self.repoint_latest()
        return {'success': True, 'message': f'deleted "{name}"'}

    def repoint_latest(self):
        """Keep latest.yaml on an existing map: the newest one left, or none."""
        latest = os.path.join(self.map_dir, 'latest.yaml')
        maps = self.list_maps()
        if any(m['latest'] for m in maps):
            return
        if not maps:
            if os.path.exists(latest):
                os.remove(latest)
            return
        src = os.path.join(self.map_dir, maps[0]['name'])
        with open(src + '.yaml') as f:
            lines = [f'image: {src}.pgm\n' if l.startswith('image:') else l for l in f]
        with open(latest, 'w') as f:
            f.writelines(lines)

    def set_pose(self, x, y, yaw):
        if self.mode != 'navigating':
            raise RuntimeError('Set pose is for navigating on a saved map (mapping knows where it is)')
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.covariance[0] = msg.pose.covariance[7] = 0.25
        msg.pose.covariance[35] = 0.07
        pose = msg.pose.pose
        pose.position.x, pose.position.y = float(x), float(y)
        pose.orientation.z, pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
        self.pose_pub.publish(msg)


def make_handler(node, web_dir):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, body, ctype='application/json'):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            node.touch('web page open')
            path = self.path.split('?')[0]
            if path in ('/', '/index.html', '/encoders', '/calibrate', '/mapedit', '/imu'):
                page = {'/encoders': 'encoders.html', '/calibrate': 'calibrate.html',
                        '/mapedit': 'mapedit.html', '/imu': 'imu.html'}.get(path, 'index.html')
                with open(os.path.join(web_dir, page), 'rb') as f:
                    self.send(200, f.read(), 'text/html; charset=utf-8')
            elif path == '/api/wheels':
                self.send(200, node.wheels_payload())
            elif path == '/api/stereo':
                self.send(200, node.stereo_state())
            elif path == '/api/stereo/sessions':
                self.send(200, {'sessions': StereoSession.sessions(node.stereo.root)})
            elif path == '/api/stereo/rectified.jpg':
                jpg = node.stereo_rectified('fresh' in self.path)
                self.send(200, jpg, 'image/jpeg') if jpg else self.send(404, {'error': 'calibrate first'})
            elif path == '/api/maps':
                self.send(200, {'maps': node.list_maps()})
            elif path.startswith('/api/mapedit/') and path.endswith(('.png', '.json')):
                name = path[len('/api/mapedit/'):].rsplit('.', 1)[0]
                try:
                    if name.endswith('.keepout') and path.endswith('.png'):
                        self.send(200, node.map_edit_keepout_png(name[:-len('.keepout')]), 'image/png')
                    elif path.endswith('.png'):
                        self.send(200, node.map_edit_png(name), 'image/png')
                    else:
                        self.send(200, node.map_edit_info(name))
                except RuntimeError as e:
                    self.send(404, {'error': str(e)})
            elif path.startswith('/api/maps/') and path.endswith('.png'):
                png = node.map_thumb(path[len('/api/maps/'):-4])
                self.send(200, png, 'image/png') if png else self.send(404, {'error': 'no such map'})
            elif path == '/api/state':
                self.send(200, node.state())
            elif path == '/api/map':
                body = node.map_payload()
                self.send(200, body) if body else self.send(404, {'error': 'no map'})
            elif path == '/api/scan':
                self.send(200, node.scan_payload())
            elif path == '/api/imu':
                since = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get('since', ['0'])[0]
                self.send(200, node.imu_payload(int(since) if since.isdigit() else 0))
            elif path == '/api/costmap':
                query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
                self.send(200, node.costmap_payload({k: int(v[0]) for k, v in query.items() if v[0].isdigit()}))
            elif path.startswith('/stream/') and path[8:] in CAMERAS:
                node.stream(self, path[8:])
            elif path == '/api/calib':
                calib = node.read_calib()
                self.send(200, calib) if calib else self.send(503, {'error': 'pico_bridge not running'})
            else:
                self.send(404, {'error': 'not found'})

        def do_POST(self):
            node.touch('web page open')
            length = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(length) or b'{}')
                path = self.path.split('?')[0]
                if path == '/api/drive':
                    node.test_abort.set()
                    node.drive(float(body.get('v', 0)), float(body.get('w', 0)), manual=True)
                    self.send(200, {'ok': True})
                elif path.startswith('/api/services/') and path[14:] in SERVICE_MENU:
                    node.set_service(path[14:], bool(body.get('on')))
                    self.send(200, {'ok': True})
                elif path == '/api/crash/remove':
                    node.crash_remove_pub.publish(Point(x=float(body.get('x', math.nan)), y=float(body.get('y', math.nan))))
                    self.send(200, {'ok': True})
                elif path == '/api/guard/override':
                    node.set_guard_override(bool(body.get('on')))
                    self.send(200, {'ok': True})
                elif path == '/api/wheels':
                    node.drive_wheels(body.get('left', 0), body.get('right', 0))
                    self.send(200, {'ok': True})
                elif path == '/api/halt':
                    node.test_abort.set()
                    node.cancel_goal()
                    node.halt()
                    self.send(200, {'ok': True})
                elif path.startswith('/api/nav/') and path[9:] in NAV_COMMANDS:
                    self.send(200, node.call_nav(path[9:]))
                elif path == '/api/goal':
                    node.send_goal(body['x'], body['y'], body['yaw'])
                    self.send(200, {'ok': True})
                elif path == '/api/initialpose':
                    node.set_pose(body['x'], body['y'], body['yaw'])
                    self.send(200, {'ok': True})
                elif path == '/api/stereo/config':
                    node.stereo.configure(body['cols'], body['rows'], body['square_mm'])
                    self.send(200, {'ok': True})
                elif path == '/api/stereo/auto':
                    node.stereo_auto = bool(body.get('on'))
                    self.send(200, {'ok': True})
                elif path == '/api/stereo/capture':
                    self.send(200, {'message': node.stereo_capture()})
                elif path == '/api/stereo/load':
                    sessions = StereoSession.sessions(node.stereo.root)
                    name = str(body.get('session') or (sessions[0]['name'] if sessions else ''))
                    if name not in [x['name'] for x in sessions]:
                        raise RuntimeError('no such calibration session')
                    with node.stereo_lock:
                        msg = node.stereo.load(os.path.join(node.stereo.root, name))
                    node.stereo_job = {'status': 'idle', 'message': '', 'summary': None}
                    node.stereo_msg = msg
                    self.send(200, {'message': msg})
                elif path == '/api/stereo/undo':
                    node.stereo.undo()
                    self.send(200, {'ok': True})
                elif path == '/api/stereo/clear':
                    node.stereo.reset()
                    self.send(200, {'ok': True})
                elif path == '/api/stereo/compute':
                    node.stereo_compute()
                    self.send(200, {'ok': True})
                elif path == '/api/stereo/save':
                    self.send(200, node.stereo_save())
                elif path == '/api/maps/save':
                    self.send(200, node.save_map(str(body.get('name', ''))))
                elif path == '/api/maps/navigate':
                    self.send(200, node.navigate_map(str(body.get('name', ''))))
                elif path == '/api/mapedit/save':
                    keepout = body.get('keepout')
                    self.send(200, node.map_edit_save(str(body['name']), str(body['png']),
                                                      None if keepout is None else str(keepout)))
                elif path == '/api/mapedit/revert':
                    self.send(200, node.map_edit_revert(str(body['name'])))
                elif path == '/api/maps/continue':
                    self.send(200, node.continue_map(str(body.get('name', ''))))
                elif path == '/api/maps/delete':
                    self.send(200, node.delete_map(str(body.get('name', ''))))
                elif path.startswith('/api/stack/') and path[11:] in STACK_ACTIONS:
                    self.send(200, node.stack(path[11:]))
                elif path.startswith('/api/test/'):
                    node.start_test(path[10:])
                    self.send(200, {'ok': True})
                elif path == '/api/calib':
                    self.send(200, node.apply_calib(body))
                elif path == '/api/lights':
                    if 'color' in body:
                        self.send(200, node.set_light_color(body.get('zone'), body.get('color')))
                    elif 'zone' in body:
                        self.send(200, node.set_light_zone(body.get('zone'), body.get('mode')))
                    else:
                        self.send(200, node.set_headlights(body.get('headlights')))
                else:
                    self.send(404, {'error': 'not found'})
            except Exception as e:
                self.send(400, {'error': str(e)})

    return Handler


def main():
    rclpy.init()
    node = WebUI()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    def spin():
        # executor.spin() ends on the first exception a callback raises, which silently
        # stopped all ROS I/O (driving included) while the web server kept answering.
        while rclpy.ok():
            try:
                executor.spin_once(timeout_sec=0.5)
            except Exception as e:  # noqa: BLE001 - log and keep the robot drivable
                if rclpy.ok():
                    node.get_logger().error(f'ROS callback failed, continuing: {e!r}', throttle_duration_sec=5)
    threading.Thread(target=spin, daemon=True).start()
    node.read_calib()

    web_dir = os.path.join(get_package_share_directory('robot_bringup'), 'web')
    server = ThreadingHTTPServer(('0.0.0.0', node.port), make_handler(node, web_dir))
    server.daemon_threads = True
    node.get_logger().info(f'web UI on http://0.0.0.0:{node.port}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        node.test_abort.set()
        node.halt()
        time.sleep(0.2)
        server.server_close()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
