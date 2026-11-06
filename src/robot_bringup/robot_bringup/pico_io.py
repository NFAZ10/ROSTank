"""Bridge to the sensor + LED Pico (firmware/pico_io, /dev/pico_io).

Sensors:
  range/front   sensor_msgs/Range: VL53L0X, ~30 Hz. +inf = nothing within range (REP 117);
                nothing is published while the sensor is missing.
  temperature   sensor_msgs/Temperature, every 2 s (DHT11)
  humidity      sensor_msgs/RelativeHumidity (0-1), every 2 s
  imu/raw       sensor_msgs/Imu: MPU6050-type chip, 50 Hz, chip axes, gyro bias NOT removed
                (mpu_imu.py does that and publishes /imu/data)
  /charger/voltage    std_msgs/Float32: volts at the charge port (charger side of its diode), 5 Hz
  /charger/connected  std_msgs/Bool, latched: above charger_on_volts, off again below
                      charger_off_volts (needs charger_count readings in a row either way)

Lights: LED groups 0 headlights (GP18), 1 body strips (GP20), 2 spare, 3 front bar (GP19),
driven from the robot's state:
  headlights    /lights/headlights (std_msgs/String 'off' | 'dim' | 'on'). 'on' is the white
                chip at 255, which is steady: dimmer levels are PWM and band in the cameras.
  body strips   one chain of side, back and side strips (body_order says which comes first).
    back        tail light: dim red; bright red braking; white reversing (from /cmd_vel_safe)
    sides       dim white idle; amber blink on the side the robot is turning to;
                battery low = orange breathe
                battery critical = every strip blinking red
  front bar     dark (the headlights light the way)
  warning       while mapping, exploring, navigating or docking (/dock/active): the robot may
                move on its own:
                front bar and side strips flash orange / white (halves swapping, over any
                turn signal), back strip wig-wag amber (brake / reverse still show)
  zone modes    /lights/zone (String '<sides|back|front> <auto|off|on>'): auto = the above,
                off = dark, on = steady in the zone's colour
  colours       /lights/color (String '<headlights|sides|back|front> #rrggbb'): the colour for
                headlights dim/on and for a zone's 'on' (default white, back red). The white part
                goes to the W chip, so pure white stays steady (no PWM) at full brightness.
Status, latched on /lights/status (String, JSON):
  {"connected", "headlights", "sides", "back", "front", "colors": {zone: "#rrggbb"}}.
If this node stops, the Pico blanks everything but a dim red breathe on the tail lights.
"""
import json
import math
import threading
import time

import rclpy
import serial
from geometry_msgs.msg import Twist
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, Range, RelativeHumidity, Temperature
from std_msgs.msg import Bool, Float32, String

HEAD, BODY, FRONT = 0, 1, 3
OFF, SOLID, BLINK, BREATHE, CHASE, WIGWAG, WIGWAG2 = range(7)
FIRMWARE_VERSION = '8'
GYRO_PER_COUNT = math.radians(1.0 / 65.5)  # +-500 deg/s range
ACCEL_PER_COUNT = 9.80665 / 4096.0         # +-8 g range
HEADLIGHT_LEVELS = {'off': 0, 'dim': 60, 'on': 255}
IDLE_SIDE = (0, 0, 0, 12)  # (r, g, b, w), dim: an indicator, not a light
AMBER = (255, 90, 0, 0)
WHITE_ON = (0, 0, 0, 255)
ORANGE_WHITE = (255, 90, 0, 200)  # WIGWAG2: halves swap between the colour and w
WARNING_MODES = ('mapping', 'exploring', 'navigating')
ZONES = ('sides', 'back', 'front')
ZONE_MODES = ('auto', 'off', 'on')
DEFAULT_COLORS = {'headlights': '#ffffff', 'sides': '#ffffff', 'back': '#ff0000', 'front': '#ffffff'}


def rgbw(color, scale=1.0):
    """'#rrggbb' -> (r, g, b, w) with the common white part on the W chip."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    w = min(r, g, b)
    return tuple(round(c * scale) for c in (r - w, g - w, b - w, w))


def valid_color(color):
    return len(color) == 7 and color[0] == '#' and all(c in '0123456789abcdefABCDEF' for c in color[1:])


def imu_from_counts(counts):
    """Firmware's gx gy gz ax ay az -> gyro rad/s, accel m/s^2."""
    return tuple(c * GYRO_PER_COUNT for c in counts[:3]), tuple(c * ACCEL_PER_COUNT for c in counts[3:])


class PicoIO(Node):
    def __init__(self):
        super().__init__('pico_io')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.port = p('port', '/dev/pico_io')
        # Per group [headlights GP18, body strips GP20, spare GP16, front bar GP19]: LED count and
        # colour order (0 GRB, 1 RGB, 2 RGBW, 3 GRBW).
        self.counts = list(p('group_counts', [14, 24, 0, 8]))
        self.types = list(p('group_types', [3, 3, 0, 0]))
        self.front_group = p('front_group', FRONT)
        # The body strips' order along the chain from the data input; equal thirds.
        self.body_order = list(p('body_order', ['right', 'back', 'left']))
        if sorted(self.body_order) != ['back', 'left', 'right']:
            raise ValueError("body_order must name 'right', 'back' and 'left' once each")
        # Groups that show the headlights (the spare can stand in for them).
        self.headlight_groups = list(p('headlight_groups', [HEAD]))
        # Extra groups that show the warning pattern instead of their usual look.
        self.warning_groups = list(self.declare_parameter(  # typed: an empty-list default reads as bytes
            'warning_groups', [], ParameterDescriptor(dynamic_typing=True)).value)
        self.headlights = p('headlights', 'off')
        self.zones = dict.fromkeys(ZONES, 'auto')
        self.colors = dict(DEFAULT_COLORS)
        self.range_frame = p('range_frame', 'range_front')
        self.imu_frame = p('imu_frame', 'mpu_imu')
        self.imu_id = None
        # Charge port sense on GP26: 6.8k / 1.5k divider, like the drive Pico's battery gauge.
        self.charger_divider = p('charger_divider', 5.533)
        self.charger_on, self.charger_off = p('charger_on_volts', 9.0), p('charger_off_volts', 6.0)
        self.charger_count = p('charger_count', 2)  # 200 ms apart
        self.charger_connected, self.charger_streak = None, 0

        self.range_pub = self.create_publisher(Range, 'range/front', 10)
        self.temp_pub = self.create_publisher(Temperature, 'temperature', 10)
        self.humidity_pub = self.create_publisher(RelativeHumidity, 'humidity', 10)
        self.imu_pub = self.create_publisher(Imu, 'imu/raw', qos_profile_sensor_data)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/lights/status', latched)
        self.charger_v_pub = self.create_publisher(Float32, '/charger/voltage', 10)
        self.charger_pub = self.create_publisher(Bool, '/charger/connected', latched)
        self.create_subscription(String, '/lights/headlights', self.on_headlights, 10)
        self.create_subscription(String, '/lights/zone', self.on_zone, 10)
        self.create_subscription(String, '/lights/color', self.on_color, 10)
        self.create_subscription(Twist, '/cmd_vel_safe', self.on_cmd, 10)
        self.create_subscription(String, '/robot/nav_mode', lambda m: setattr(self, 'mode', m.data), latched)
        self.create_subscription(String, '/battery/status', lambda m: setattr(self, 'battery', m.data), latched)
        self.docking = False
        self.create_subscription(Bool, '/dock/active', lambda m: setattr(self, 'docking', m.data), latched)

        self.mode, self.battery = 'idle', ''
        self.v = self.w = 0.0
        self.cmd_time = 0.0
        self.brake_until = 0.0
        self.sent = {}           # (group, first, n) -> line last sent
        self.last_full = 0.0
        self.ser = None
        self.ser_lock = threading.Lock()
        self.connected = False
        self.publish_status()
        self.create_timer(0.1, self.update)
        self.create_timer(2.0, self.publish_status)  # heartbeat: the web UI treats stale as down
        threading.Thread(target=self.reader, daemon=True).start()

    # ---------- inputs ----------
    def on_headlights(self, msg):
        if msg.data in HEADLIGHT_LEVELS and msg.data != self.headlights:
            self.headlights = msg.data
            self.publish_status()

    def on_zone(self, msg):
        parts = msg.data.split()
        if len(parts) == 2 and parts[0] in ZONES and parts[1] in ZONE_MODES and self.zones[parts[0]] != parts[1]:
            self.zones[parts[0]] = parts[1]
            self.publish_status()

    def on_color(self, msg):
        parts = msg.data.split()
        if len(parts) == 2 and parts[0] in self.colors and valid_color(parts[1]):
            self.colors[parts[0]] = parts[1].lower()
            self.publish_status()

    def on_cmd(self, msg):
        now = time.monotonic()
        v = msg.linear.x
        if abs(v) < abs(self.v) - 0.02 or (abs(self.v) > 0.02 and abs(v) <= 0.01):
            self.brake_until = now + 0.8
        self.v, self.w, self.cmd_time = v, msg.angular.z, now

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps(
            {'connected': self.connected, 'headlights': self.headlights, **self.zones,
             'colors': self.colors})))

    # ---------- lights ----------
    def segments(self):
        """Desired look: {(group, first, n): (effect, (r, g, b, w), period_ms)}."""
        now = time.monotonic()
        if now - self.cmd_time > 0.5:  # commands stopped: a stop, so brake if it was moving
            if abs(self.v) > 0.02:
                self.brake_until = now + 0.8
            self.v = self.w = 0.0
        out = {}
        level = HEADLIGHT_LEVELS.get(self.headlights, 0)
        for g in self.headlight_groups:
            if self.counts[g]:
                out[(g, 0, self.counts[g])] = (SOLID if level else OFF, rgbw(self.colors['headlights'], level / 255), 1000)

        warning = self.mode in WARNING_MODES or self.docking
        n = self.counts[BODY]
        if n:
            critical = self.battery.startswith(('critical', 'shutting'))
            if self.v < -0.02:
                back = (SOLID, WHITE_ON, 1000)
            elif now < self.brake_until:
                back = (SOLID, (255, 0, 0, 0), 1000)
            elif warning:
                back = (WIGWAG, AMBER, 1000)
            else:
                back = (SOLID, (50, 0, 0, 0), 1000)
            side = (SOLID, IDLE_SIDE, 1000)
            if self.battery.startswith('low'):
                side = (BREATHE, (255, 60, 0, 0), 2000)
            look = {'back': back, 'left': side, 'right': side}
            if warning:  # like hazard lights: over the turn signals
                look['left'] = look['right'] = (WIGWAG2, ORANGE_WHITE, 800)
            elif abs(self.w) > 0.3:  # turn signal on the side it's turning towards
                look['left' if (self.w > 0) == (self.v >= 0) else 'right'] = (BLINK, AMBER, 700)
            if critical:
                look = dict.fromkeys(look, (BLINK, (255, 0, 0, 0), 500))
            for name in look:
                zone = 'back' if name == 'back' else 'sides'
                if self.zones[zone] == 'off':
                    look[name] = (OFF, (0, 0, 0, 0), 1000)
                elif self.zones[zone] == 'on':
                    look[name] = (SOLID, rgbw(self.colors[zone]), 1000)
            third = n // 3
            for i, name in enumerate(self.body_order):
                first = i * third
                out[(BODY, first, third if i < 2 else n - first)] = look[name]

        g = self.front_group
        if 0 <= g < len(self.counts) and self.counts[g] and g not in self.headlight_groups:
            mode = self.zones['front']
            if mode == 'on':
                look = (SOLID, rgbw(self.colors['front']), 1000)
            elif mode == 'auto' and warning:
                look = (WIGWAG2, ORANGE_WHITE, 800)
            else:
                look = (OFF, (0, 0, 0, 0), 1000)
            out[(g, 0, self.counts[g])] = look
        for g in self.warning_groups:
            if self.counts[g] and warning:
                out = {k: v for k, v in out.items() if k[0] != g}
                out[(g, 0, self.counts[g])] = (WIGWAG, AMBER, 1000)
        return out

    def update(self):
        if not self.connected:
            return
        now = time.monotonic()
        full = now - self.last_full > 3.0  # repeat everything now and then, in case a line was lost
        if full:
            self.last_full = now
        segments = self.segments()
        if set(segments) != set(self.sent):  # split changed (e.g. warning beacon <-> 3 strips): send all
            self.sent.clear()
        lines = []
        for (g, first, n), (effect, (r, gr, b, w), period) in segments.items():
            line = f'L {g} {first} {n} {effect} {r} {gr} {b} {w} {period}'
            if full or self.sent.get((g, first, n)) != line:
                self.sent[(g, first, n)] = line
                lines.append(line)
        self.write('\n'.join(lines + ['K']) + '\n')

    def configure(self):
        """Set up every group; the Pico forgets it all on a reset."""
        self.sent.clear()
        self.last_full = 0.0
        self.write(''.join(f'C {g} {self.counts[g]} {self.types[g]}\n' for g in range(4)))

    # ---------- serial ----------
    def write(self, text):
        with self.ser_lock:
            if self.ser is None:
                return
            try:
                self.ser.write(text.encode())
            except serial.SerialException:
                self.drop_serial()

    def drop_serial(self):
        try:
            self.ser.close()
        except Exception:
            pass
        self.ser = None
        if self.connected:
            self.connected = False
            self.publish_status()

    def reader(self):
        while rclpy.ok():
            if self.ser is None:
                try:
                    s = serial.Serial(self.port, 115200, timeout=0.5)
                    with self.ser_lock:
                        self.ser = s
                    self.get_logger().info(f'connected to {self.port}')
                    self.configure()
                    self.connected = True
                    self.publish_status()
                except serial.SerialException:
                    time.sleep(1.0)
                    continue
            try:
                raw = self.ser.readline()
            except (serial.SerialException, AttributeError, TypeError):
                self.get_logger().warn(f'lost {self.port}, reconnecting')
                with self.ser_lock:
                    self.drop_serial()
                continue
            try:
                self.on_line(raw.decode(errors='ignore').split())
            except ValueError:
                pass

    def on_charger(self, volts):
        self.charger_v_pub.publish(Float32(data=volts))
        before = self.charger_connected
        self.charger_connected, self.charger_streak = charger_state(
            before, self.charger_streak, volts, self.charger_on, self.charger_off, self.charger_count)
        if self.charger_connected != before:
            self.charger_pub.publish(Bool(data=self.charger_connected))
            self.get_logger().info(f'charger {"connected" if self.charger_connected else "not connected"} ({volts:.1f} V)')

    def on_line(self, parts):
        if len(parts) == 7 and parts[0] == 'I':
            gyro, accel = imu_from_counts([int(c) for c in parts[1:]])
            msg = Imu()
            msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), self.imu_frame
            msg.orientation_covariance[0] = -1.0  # no orientation (REP 145)
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = gyro
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = accel
            self.imu_pub.publish(msg)
        elif len(parts) == 2 and parts[0] == 'V':
            self.on_charger(int(parts[1]) / 1000.0 * self.charger_divider)
        elif len(parts) == 2 and parts[0] == 'M':
            if parts[1] != self.imu_id:
                self.imu_id = parts[1]
                self.get_logger().info('IMU not answering' if parts[1] == '-1' else f'IMU chip id 0x{int(parts[1]):02x}')
        elif len(parts) == 3 and parts[0] == 'R' and parts[1] != '-1':
            msg = Range()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = self.range_frame
            msg.radiation_type = Range.INFRARED
            msg.field_of_view = math.radians(25.0)
            msg.min_range, msg.max_range = 0.03, 2.0
            mm, ok = int(parts[1]), parts[2] == '1'
            msg.range = mm / 1000.0 if ok and mm / 1000.0 <= msg.max_range else math.inf
            self.range_pub.publish(msg)
        elif len(parts) == 3 and parts[0] == 'D' and parts[1] != 'nan':
            stamp = self.get_clock().now().to_msg()
            t = Temperature(temperature=float(parts[1]), variance=1.0)  # DHT11: +-2 C
            t.header.stamp, t.header.frame_id = stamp, 'base_link'
            h = RelativeHumidity(relative_humidity=float(parts[2]) / 100.0, variance=0.0025)
            h.header.stamp, h.header.frame_id = stamp, 'base_link'
            self.temp_pub.publish(t)
            self.humidity_pub.publish(h)
        elif len(parts) == 3 and parts[0] == 'N' and parts[1] == 'pico_io' and parts[2] != FIRMWARE_VERSION:
            self.get_logger().warn(f'pico_io firmware version {parts[2]}, expected {FIRMWARE_VERSION}: '
                                   'robot flash io', once=True)


def charger_state(connected, streak, volts, on, off, count):
    """Debounced charger detection: (connected, streak) after one reading."""
    want = volts >= on if not connected else volts > off
    if connected is None:
        return want, 0
    if want == connected:
        return connected, 0
    streak += 1
    return (want, 0) if streak >= count else (connected, streak)


def main():
    rclpy.init()
    node = PicoIO()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
