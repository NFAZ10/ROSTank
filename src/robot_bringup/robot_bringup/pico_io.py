"""Bridge to the sensor + LED Pico (firmware/pico_io, /dev/pico_io).

Sensors:
  range/front   sensor_msgs/Range: VL53L0X, ~30 Hz. +inf = nothing within range (REP 117);
                nothing is published while the sensor is missing.
  temperature   sensor_msgs/Temperature, every 2 s (DHT11)
  humidity      sensor_msgs/RelativeHumidity (0-1), every 2 s

Lights: the four LED groups (headlights, top strips, tail lights, spare) are driven from
the robot's state:
  headlights    /lights/headlights (std_msgs/String 'off' | 'dim' | 'on'). 'on' is the white
                chip at 255, which is steady: dimmer levels are PWM and band in the cameras.
  tail lights   dim red; bright red braking; white reversing (from /cmd_vel_safe)
  top strips    dim white idle; amber blink on the side the robot is turning to; battery
                low = back strip orange, critical = all red blinking
  warning       while mapping, exploring or navigating (the robot may move on its own):
                amber beacon running round the top strips, tail rings wig-wag amber
                (brake / reverse still show), and the warning_groups wig-wag amber too
Status, latched on /lights/status (String, JSON): {"connected", "headlights"}.
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
from sensor_msgs.msg import Range, RelativeHumidity, Temperature
from std_msgs.msg import String

HEAD, TOP, TAIL, SPARE = 0, 1, 2, 3
OFF, SOLID, BLINK, BREATHE, CHASE, WIGWAG = range(6)
FIRMWARE_VERSION = '2'
HEADLIGHT_LEVELS = {'off': 0, 'dim': 60, 'on': 255}
IDLE_TOP = (0, 0, 0, 12)  # (r, g, b, w), dim: an indicator, not a light
AMBER = (255, 90, 0, 0)
WARNING_MODES = ('mapping', 'exploring', 'navigating')


class PicoIO(Node):
    def __init__(self):
        super().__init__('pico_io')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.port = p('port', '/dev/pico_io')
        # Per group [headlights, top, tails, spare]: LED count and colour order
        # (0 GRB, 1 RGB, 2 RGBW, 3 GRBW). Top = left, back, right strips chained in that order;
        # headlights and tails = left ring then right ring.
        self.counts = list(p('group_counts', [16, 24, 16, 0]))
        self.types = list(p('group_types', [3, 3, 3, 0]))
        # Groups that show the headlights (the spare can stand in for them).
        self.headlight_groups = list(p('headlight_groups', [HEAD]))
        # Extra groups that show the warning pattern instead of their usual look.
        self.warning_groups = list(self.declare_parameter(  # typed: an empty-list default reads as bytes
            'warning_groups', [], ParameterDescriptor(dynamic_typing=True)).value)
        self.headlights = p('headlights', 'off')
        self.range_frame = p('range_frame', 'range_front')

        self.range_pub = self.create_publisher(Range, 'range/front', 10)
        self.temp_pub = self.create_publisher(Temperature, 'temperature', 10)
        self.humidity_pub = self.create_publisher(RelativeHumidity, 'humidity', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/lights/status', latched)
        self.create_subscription(String, '/lights/headlights', self.on_headlights, 10)
        self.create_subscription(Twist, '/cmd_vel_safe', self.on_cmd, 10)
        self.create_subscription(String, '/robot/nav_mode', lambda m: setattr(self, 'mode', m.data), latched)
        self.create_subscription(String, '/battery/status', lambda m: setattr(self, 'battery', m.data), latched)

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

    def on_cmd(self, msg):
        now = time.monotonic()
        v = msg.linear.x
        if abs(v) < abs(self.v) - 0.02 or (abs(self.v) > 0.02 and abs(v) <= 0.01):
            self.brake_until = now + 0.8
        self.v, self.w, self.cmd_time = v, msg.angular.z, now

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps(
            {'connected': self.connected, 'headlights': self.headlights})))

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
                out[(g, 0, self.counts[g])] = (SOLID if level else OFF, (0, 0, 0, level), 1000)

        warning = self.mode in WARNING_MODES
        if self.counts[TAIL]:
            if self.v < -0.02:
                look = (SOLID, (0, 0, 0, 255), 1000)
            elif now < self.brake_until:
                look = (SOLID, (255, 0, 0, 0), 1000)
            elif warning:
                look = (WIGWAG, AMBER, 1000)
            else:
                look = (SOLID, (50, 0, 0, 0), 1000)
            out[(TAIL, 0, self.counts[TAIL])] = look

        n = self.counts[TOP]
        if n:
            third = n // 3
            left = back = right = (SOLID, IDLE_TOP, 1000)
            if self.battery.startswith('low'):
                back = (BREATHE, (255, 60, 0, 0), 2000)
            if abs(self.w) > 0.3:  # turn signal on the side it's turning towards
                if (self.w > 0) == (self.v >= 0):
                    left = (BLINK, AMBER, 700)
                else:
                    right = (BLINK, AMBER, 700)
            if self.battery.startswith(('critical', 'shutting')):
                left = back = right = (BLINK, (255, 0, 0, 0), 500)
            if warning and not self.battery.startswith(('critical', 'shutting')):
                out[(TOP, 0, n)] = (CHASE, AMBER, 1200)  # one beacon all the way round
            else:
                out[(TOP, 0, third)] = left
                out[(TOP, third, third)] = back
                out[(TOP, 2 * third, n - 2 * third)] = right
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

    def on_line(self, parts):
        if len(parts) == 3 and parts[0] == 'R' and parts[1] != '-1':
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
