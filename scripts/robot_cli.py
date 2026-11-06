"""ROS side of the `robot` command (see scripts/robot). Run through it, not directly.

Each subcommand is a short-lived rclpy node talking to the running stack:
  status-topics | move <m> [speed] | turn <deg> [rate] | stop | goal <x> <y> [yaw_deg]
  pose <x> <y> [yaw_deg] | where | wheels | snap [left|right] | calib [show | set k=v ...]
  maps | map-delete <name>
"""
import glob
import math
import os
import re
import signal
import sys
import time

import rclpy
from action_msgs.srv import CancelGoal
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
import numpy as np
from sensor_msgs.msg import BatteryState, CompressedImage, Image, JointState, LaserScan
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener

WS = os.path.expanduser('~/robot_ws')
MAP_DIR = os.path.join(WS, 'maps')
DRIVE_YAML = os.path.join(WS, 'src/robot_bringup/config/drive.yaml')
TICKS_PER_REV = 748.0  # = firmware TICKS_PER_REV
CALIB_LIVE = ['wheel_radius', 'track_width', 'invert_left', 'invert_right', 'swap_sides',
              'battery_scale', 'battery_divider']
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)


class Cli(Node):
    def __init__(self):
        super().__init__('robot_cli')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def spin_for(self, seconds, until=lambda: False):
        end = time.monotonic() + seconds
        while time.monotonic() < end and not until():
            rclpy.spin_once(self, timeout_sec=0.05)

    def latest(self, msg_type, topic, timeout=2.0, qos=10):
        box = []
        sub = self.create_subscription(msg_type, topic, box.append, qos)
        self.spin_for(timeout, lambda: box)
        self.destroy_subscription(sub)
        return box[-1] if box else None

    def nav_mode(self):
        m = self.latest(String, '/robot/nav_mode', 1.5, LATCHED)
        return m.data if m else 'off'

    def stereo(self, state):
        c = self.create_client(SetBool, '/depth/set_enabled')
        if not c.wait_for_service(timeout_sec=5.0):
            die('stereo is not responding (robot restart)')
        future = c.call_async(SetBool.Request(data=state == 'on'))
        self.spin_for(5.0, future.done)
        print(future.result().message if future.done() else 'no answer from stereo')

    def guard(self, state):
        c = self.create_client(SetBool, '/drive_guard/override')
        if not c.wait_for_service(timeout_sec=5.0):
            die('drive guard is not responding (robot restart)')
        future = c.call_async(SetBool.Request(data=state == 'off'))
        self.spin_for(5.0, future.done)
        print(future.result().message if future.done() else 'no answer from the drive guard')

    def halt(self):
        for _ in range(5):
            self.cmd_pub.publish(Twist())
            time.sleep(0.02)

    # ---------- status ----------
    def status_topics(self):
        # Not the cameras: subscribing would start them (camera_manager.py).
        topics = {'/wheel/odom': Odometry, '/odom': Odometry, '/scan': LaserScan}
        counts = {t: 0 for t in topics}
        for t, ty in topics.items():
            self.create_subscription(ty, t, lambda m, t=t: counts.__setitem__(t, counts[t] + 1),
                                     qos_profile_sensor_data)
        mode = map_name = None
        def on_mode(m):
            nonlocal mode
            mode = m.data
        def on_map(m):
            nonlocal map_name
            map_name = m.data
        self.create_subscription(String, '/robot/nav_mode', on_mode, LATCHED)
        self.create_subscription(String, '/robot/nav_map', on_map, LATCHED)
        guard = None
        def on_guard(m):
            nonlocal guard
            guard = m.data
        self.create_subscription(String, '/drive_guard/status', on_guard, LATCHED)
        cameras = None
        def on_cameras(m):
            nonlocal cameras
            cameras = m.data
        self.create_subscription(String, '/cameras/status', on_cameras, LATCHED)
        imu = None
        def on_imu(m):
            nonlocal imu
            imu = m.data
        self.create_subscription(String, '/imu/status', on_imu, LATCHED)
        batt = []
        self.create_subscription(BatteryState, '/battery', batt.append, 10)
        self.spin_for(0.5)  # discovery
        for t in counts:
            counts[t] = 0
        self.spin_for(2.0)
        label = {'/wheel/odom': 'drive (Pico)', '/odom': 'fused odometry', '/scan': 'lidar'}
        for t, n in counts.items():
            hz = n / 2.0
            state = 'ok' if hz > 1 else ('idle' if t == '/scan' else 'NO DATA')
            note = '  (motor spins only while something uses it)' if t == '/scan' and hz == 0 else ''
            print(f'  {label[t]:<16} {hz:5.1f} Hz  {state}{note}')
        print(f'  {"cameras":<16} {cameras or "unknown"}')
        print(f'  {"IMU":<16} {imu or "unknown"}')
        print(f'  {"nav mode":<16} {mode or "unknown"}' + (f'  on map "{map_name}"' if map_name else ''))
        print(f'  {"drive guard":<16} {guard or "not running"}')
        b = batt[-1] if batt else None
        print(f'  {"battery":<16} ' + (f'{b.voltage:.2f} V, about {b.percentage * 100:.0f}%' if b and b.present
                                        else 'not connected (divider on GP26)' if b else 'no reading'))

    # ---------- motion ----------
    def drive_until(self, v, w, done, timeout, label):
        mode = self.nav_mode()
        if mode in ('mapping', 'navigating', 'exploring'):
            die(f'{mode} is running: stop it first (robot map stop), Nav2 would fight this')
        odom = []
        self.create_subscription(Odometry, '/odom', odom.append, 10)
        self.spin_for(2.0, lambda: odom)
        if not odom:
            die('no /odom: is the stack running? (robot status)')
        p0 = odom[-1].pose.pose
        x0, y0, last_yaw = p0.position.x, p0.position.y, yaw_of(p0.orientation)
        turned = 0.0
        end = time.monotonic() + timeout
        print(label + '  (Ctrl-C stops)')
        try:
            while True:
                self.spin_for(0.05)
                p = odom[-1].pose.pose
                yaw = yaw_of(p.orientation)
                turned += math.atan2(math.sin(yaw - last_yaw), math.cos(yaw - last_yaw))
                last_yaw = yaw
                dist = math.hypot(p.position.x - x0, p.position.y - y0)
                if done(dist, turned):
                    break
                if time.monotonic() > end:
                    self.halt()
                    die(f'timed out after {dist:.2f} m / {math.degrees(turned):.0f} deg (blocked or slipping?)')
                t = Twist()
                t.linear.x, t.angular.z = v, w
                self.cmd_pub.publish(t)
        finally:
            self.halt()
        print(f'done: {dist:.2f} m, {math.degrees(turned):.0f} deg (by odometry)')

    def move(self, metres, speed=0.15):
        speed = min(abs(speed), 0.4)
        self.drive_until(math.copysign(speed, metres), 0.0, lambda d, a: d >= abs(metres),
                         abs(metres) / speed * 2 + 3, f'moving {metres:+.2f} m at {speed:.2f} m/s')

    def turn(self, degrees, rate=0.8):
        rate = min(abs(rate), 1.5)
        target = math.radians(degrees)
        self.drive_until(0.0, math.copysign(rate, target), lambda d, a: abs(a) >= abs(target),
                         abs(target) / rate * 2 + 3, f'turning {degrees:+.0f} deg (+ = left)')

    def stop(self):
        self.halt()
        client = self.create_client(CancelGoal, '/navigate_to_pose/_action/cancel_goal')
        if client.wait_for_service(timeout_sec=1.0):
            future = client.call_async(CancelGoal.Request())  # zero goal id + time = cancel all
            self.spin_for(2.0, future.done)
            print('motors stopped, navigation goals cancelled')
        else:
            print('motors stopped (no navigation running)')

    # ---------- navigation ----------
    def goal(self, x, y, yaw_deg=0.0):
        client = ActionClient(self, NavigateToPose, '/navigate_to_pose')
        if not client.wait_for_server(timeout_sec=3.0):
            die('navigation is not running: robot map start, or robot map nav <name>')
        tf = Buffer()
        TransformListener(tf, self)
        self.spin_for(1.0)
        if not tf.can_transform('map', 'base_link', Time()):
            die("navigation doesn't know where the robot is: robot pose <x> <y> [yaw] first")
        g = NavigateToPose.Goal()
        g.pose.header.frame_id = 'map'
        g.pose.header.stamp = self.get_clock().now().to_msg()
        g.pose.pose.position.x, g.pose.pose.position.y = x, y
        yaw = math.radians(yaw_deg)
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        last = [0.0]
        def feedback(fb):
            if time.monotonic() - last[0] > 1.0:
                last[0] = time.monotonic()
                print(f'  {fb.feedback.distance_remaining:5.2f} m to go', flush=True)
        send = client.send_goal_async(g, feedback_callback=feedback)
        self.spin_for(5.0, send.done)
        handle = send.result() if send.done() else None
        if handle is None or not handle.accepted:
            die('navigation refused the goal')
        print(f'driving to ({x:.2f}, {y:.2f}), facing {yaw_deg:.0f} deg  (Ctrl-C cancels)')
        result = handle.get_result_async()
        try:
            while not result.done():
                rclpy.spin_once(self, timeout_sec=0.1)
        except KeyboardInterrupt:
            handle.cancel_goal_async()
            self.spin_for(1.0)
            self.halt()
            die('cancelled')
        status = result.result().status
        print({4: 'reached the goal', 5: 'goal cancelled'}.get(status, "couldn't reach the goal (blocked or no path)"))

    def pose(self, x, y, yaw_deg=0.0):
        pub = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.spin_for(3.0, lambda: pub.get_subscription_count() > 0)
        if pub.get_subscription_count() == 0:
            die('nothing is listening on /initialpose: start navigation on a map first (robot map nav <name>)')
        m = PoseWithCovarianceStamped()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        yaw = math.radians(yaw_deg)
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        m.pose.covariance[0] = m.pose.covariance[7] = 0.25
        m.pose.covariance[35] = 0.07
        pub.publish(m)
        self.spin_for(0.5)
        print(f'pose set to ({x:.2f}, {y:.2f}), facing {yaw_deg:.0f} deg')

    def where(self):
        tf = Buffer()
        TransformListener(tf, self)
        self.spin_for(1.5)
        for frame in ('map', 'odom'):
            if tf.can_transform(frame, 'base_link', Time()):
                t = tf.lookup_transform(frame, 'base_link', Time()).transform
                print(f'{frame}: x {t.translation.x:+.2f} m  y {t.translation.y:+.2f} m  '
                      f'facing {math.degrees(yaw_of(t.rotation)):+.0f} deg')
                return
        die('no robot pose (is the stack running?)')

    # ---------- sensors ----------
    def wheels(self):
        state = []
        self.create_subscription(JointState, '/wheel/joint_states', state.append, 10)
        self.spin_for(2.0, lambda: state)
        if not state:
            die('no /wheel/joint_states: is the stack running?')
        base = state[-1].position
        print('counts since start, sprocket speed (rad/s), commanded  (Ctrl-C quits)')
        try:
            while True:
                self.spin_for(0.25)
                s = state[-1]
                parts = []
                for i, side in enumerate(('left', 'right')):
                    ticks = round((s.position[i] - base[i]) * TICKS_PER_REV / (2 * math.pi))
                    parts.append(f'{side} {ticks:+7d}  {s.velocity[i]:+5.2f}  ({s.effort[i]:+5.2f})')
                print('\r' + '    '.join(parts), end='', flush=True)
                del state[:-1]
        except KeyboardInterrupt:
            print()

    def battery(self):
        latest = []
        self.create_subscription(BatteryState, '/battery', latest.append, 10)
        print('pack voltage (10 s average), approximate charge  (Ctrl-C quits)')
        try:
            while True:
                self.spin_for(1.0)
                if latest:
                    b = latest[-1]
                    text = f'{b.voltage:6.2f} V  about {b.percentage * 100:3.0f}%' if b.present else 'not connected'
                    print('\r' + text + ' ' * 10, end='', flush=True)
        except KeyboardInterrupt:
            print()

    def snap(self, side='left'):
        topic = '/depth/preview/compressed' if side == 'depth' else f'/{side}/preview/image/compressed'
        msg = self.latest(CompressedImage, topic, 45.0, qos_profile_sensor_data)  # a camera may need to start
        if msg is None:
            die(f'no image from the {side} camera (robot status)')
        out_dir = os.path.expanduser('~/robot_snaps')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, time.strftime(f'{side}_%Y%m%d_%H%M%S.jpg'))
        with open(path, 'wb') as f:
            f.write(bytes(msg.data))
        print(path)

    def depth_ahead(self):
        """Distance to what's straight ahead, from stereo depth (starts it while we watch)."""
        msg = self.latest(Image, '/depth/image', 45.0, qos_profile_sensor_data)  # the cameras may need to start
        if msg is None:
            die('no depth: are both cameras running? (robot status)')
        d = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        h, w = d.shape
        for name, (x0, x1) in (('left', (0, w // 3)), ('ahead', (w // 3, 2 * w // 3)), ('right', (2 * w // 3, w))):
            band = d[h // 3:2 * h // 3, x0:x1]
            v = band[band > 0]
            near = f'nearest {np.percentile(v, 5):.2f} m, typical {np.median(v):.2f} m' if v.size > 50 else 'no depth'
            print(f'  {name:<6} {near}  ({100 * v.size / band.size:.0f}% of pixels measured)')

    # ---------- drive calibration ----------
    def param_client(self, srv, name):
        c = self.create_client(srv, f'/pico_bridge/{name}')
        if not c.wait_for_service(timeout_sec=5.0):
            die('pico_bridge is not running (robot status)')
        return c

    def calib_show(self):
        c = self.param_client(GetParameters, 'get_parameters')
        req = GetParameters.Request()
        req.names = CALIB_LIVE + ['max_wheel_speed']
        fut = c.call_async(req)
        self.spin_for(3.0, fut.done)
        for name, v in zip(req.names, fut.result().values):
            val = v.double_value if v.type == ParameterType.PARAMETER_DOUBLE else v.bool_value
            print(f'  {name:<16} {val}')

    def calib_set(self, pairs):
        req = SetParameters.Request()
        persist = {}
        for pair in pairs:
            if '=' not in pair:
                die(f'expected name=value, got "{pair}"')
            name, raw = pair.split('=', 1)
            if name not in CALIB_LIVE:
                die(f'{name}: settable are {", ".join(CALIB_LIVE)}')
            if name.startswith(('invert', 'swap')):
                val = raw.lower() in ('1', 'true', 'yes', 'on')
                pv = ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=val)
                persist[name] = 'true' if val else 'false'
            else:
                val = float(raw)
                if val <= 0:
                    die(f'{name} must be > 0')
                pv = ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=val)
                persist[name] = f'{val:.4f}'
            req.parameters.append(Parameter(name=name, value=pv))
        c = self.param_client(SetParameters, 'set_parameters')
        fut = c.call_async(req)
        self.spin_for(3.0, fut.done)
        bad = [r.reason for r in fut.result().results if not r.successful]
        if bad:
            die('refused: ' + '; '.join(bad))
        # Same in-place edit as web_ui.persist(): keep comments, replace the value.
        with open(DRIVE_YAML) as f:
            text = f.read()
        for name, literal in persist.items():
            pattern = re.compile(rf'^(\s*{name}:\s*)[^#\n]*?(\s*(#.*)?)$', re.M)
            if pattern.search(text):
                text = pattern.sub(lambda m: m.group(1) + literal + (m.group(2) or ''), text, count=1)
            else:
                text = text.rstrip('\n') + f'\n    {name}: {literal}\n'
        with open(DRIVE_YAML, 'w') as f:
            f.write(text)
        print('applied live and saved to drive.yaml: ' + ', '.join(f'{k}={v}' for k, v in persist.items()))


# ---------- map files (no ROS needed) ----------
def latest_name():
    try:
        with open(os.path.join(MAP_DIR, 'latest.yaml')) as f:
            img = next(l for l in f if l.startswith('image:')).split(':', 1)[1].strip()
        return os.path.splitext(os.path.basename(img))[0]
    except (OSError, StopIteration):
        return ''


def list_maps():
    latest = latest_name()
    names = sorted((os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(MAP_DIR, '*.yaml'))
                    if not p.endswith('/latest.yaml')), key=lambda n: -os.path.getmtime(os.path.join(MAP_DIR, n + '.yaml')))
    if not names:
        print('no saved maps (robot map start, drive around, robot map save <name>)')
    for n in names:
        when = time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(os.path.join(MAP_DIR, n + '.yaml'))))
        flags = ('  latest' if n == latest else '') + \
                ('  remembers pose' if os.path.exists(os.path.join(MAP_DIR, n + '.pose.json')) else '')
        print(f'  {n:<28} {when}{flags}')


def delete_map(name):
    if not re.match(r'^[A-Za-z0-9_-]{1,40}$', name) or name == 'latest':
        die('unknown map')
    removed = 0
    for ext in ('.yaml', '.pgm', '.pose.json'):
        p = os.path.join(MAP_DIR, name + ext)
        if os.path.exists(p):
            os.remove(p)
            removed += 1
    if not removed:
        die(f'no map called {name}')
    if latest_name() == name:  # point latest at the newest map left, like the web UI
        rest = sorted(glob.glob(os.path.join(MAP_DIR, '*.yaml')), key=os.path.getmtime)
        rest = [p for p in rest if not p.endswith('/latest.yaml')]
        latest = os.path.join(MAP_DIR, 'latest.yaml')
        if rest:
            src = rest[-1][:-5]
            with open(src + '.yaml') as f:
                lines = [f'image: {src}.pgm\n' if l.startswith('image:') else l for l in f]
            with open(latest, 'w') as f:
                f.writelines(lines)
        elif os.path.exists(latest):
            os.remove(latest)
    print(f'deleted {name}')


def main():
    args = sys.argv[1:]
    if not args:
        die(__doc__)
    cmd, rest = args[0], args[1:]
    if cmd == 'maps':
        return list_maps()
    if cmd == 'map-delete':
        return delete_map(rest[0]) if rest else die('usage: robot map delete <name>')
    rclpy.init()
    node = Cli()
    # Ctrl-C: let the command's own cleanup (zero velocity) run.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        f = [float(a) for a in rest if re.match(r'^[-+]?[0-9.]+$', a)]
        if cmd == 'status-topics':
            node.status_topics()
        elif cmd == 'move':
            node.move(*f[:2]) if f else die('usage: robot move <metres> [speed m/s]   (negative = backwards)')
        elif cmd == 'turn':
            node.turn(*f[:2]) if f else die('usage: robot turn <degrees> [rate rad/s]   (positive = left)')
        elif cmd == 'stop':
            node.stop()
        elif cmd == 'stereo':
            node.stereo(rest[0]) if rest and rest[0] in ('on', 'off') else die('usage: robot stereo off | on')
        elif cmd == 'guard':
            node.guard(rest[0]) if rest and rest[0] in ('on', 'off') else die('usage: robot guard off | on')
        elif cmd == 'goal':
            node.goal(*f[:3]) if len(f) >= 2 else die('usage: robot goal <x> <y> [facing deg]')
        elif cmd == 'pose':
            node.pose(*f[:3]) if len(f) >= 2 else die('usage: robot pose <x> <y> [facing deg]')
        elif cmd == 'where':
            node.where()
        elif cmd == 'wheels':
            node.wheels()
        elif cmd == 'snap':
            node.snap(rest[0] if rest and rest[0] in ('left', 'right', 'depth') else 'left')
        elif cmd == 'depth':
            node.depth_ahead()
        elif cmd == 'battery':
            node.battery()
        elif cmd == 'calib':
            if not rest or rest[0] == 'show':
                node.calib_show()
            elif rest[0] == 'set' and len(rest) > 1:
                node.calib_set(rest[1:])
            else:
                die('usage: robot calib [show] | robot calib set wheel_radius=0.031 track_width=0.19 ...')
        else:
            die(f'unknown command {cmd}')
    except KeyboardInterrupt:
        node.halt()
        print('\nstopped')
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
