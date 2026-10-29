"""Battery guard: warns on a low pack, and shuts the Jetson down cleanly before it goes flat.

Reads /battery (pico_bridge). Latched status on /battery/status:
  'not connected' | 'ok' | 'low: charge soon' | 'critical: charge now' | 'shutting down'
Low = under low_voltage for 30 s, critical = under critical_voltage for 30 s.
Below shutdown_voltage for 60 s: `sudo -n poweroff`, before the packs' own protection cuts the
power and risks the Jetson's filesystem. Only armed after 20 s of plausible readings
(arm_voltage..15.5 V), and reset whenever the divider reads as not connected, so a loose or
floating wire can't trigger it.
shutdown_voltage 0 disables the shutdown; dry_run logs instead of powering off.
"""
import subprocess
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import BatteryState
from std_msgs.msg import String


class BatteryGuard(Node):
    def __init__(self):
        super().__init__('battery_guard')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.low = p('low_voltage', 12.4)
        self.critical = p('critical_voltage', 12.0)
        self.shutdown = p('shutdown_voltage', 11.0)
        self.arm_voltage = p('arm_voltage', 11.5)
        self.low_s, self.shutdown_s = p('low_seconds', 30.0), p('shutdown_seconds', 60.0)
        self.dry_run = p('dry_run', False)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/battery/status', latched)
        self.create_subscription(BatteryState, '/battery', self.on_battery, 10)
        self.armed = False
        self.plausible_since = None
        self.below = {}  # threshold name -> monotonic time it went below
        self.status = None
        self.done = False
        self.say('not connected')

    def say(self, text):
        if text != self.status:
            self.status = text
            self.status_pub.publish(String(data=text))
            self.get_logger().info(f'battery: {text}')

    def held(self, name, condition, seconds):
        """True once condition has held continuously for seconds."""
        now = time.monotonic()
        if not condition:
            self.below.pop(name, None)
            return False
        start = self.below.setdefault(name, now)
        return now - start >= seconds

    def on_battery(self, msg):
        if self.done:
            return
        if not msg.present:
            self.below.clear()
            self.armed, self.plausible_since = False, None
            self.say('not connected')
            return
        v = msg.voltage
        if self.arm_voltage <= v <= 15.5:
            self.plausible_since = self.plausible_since or time.monotonic()
            if time.monotonic() - self.plausible_since >= 20.0 and not self.armed:
                self.armed = True
                self.get_logger().info('low-battery shutdown armed')
        else:
            self.plausible_since = None
        if self.shutdown > 0 and self.armed and self.held('shutdown', v < self.shutdown, self.shutdown_s):
            self.say('shutting down')
            self.done = True
            if self.dry_run:
                self.get_logger().warn(f'battery {v:.2f} V: would power off now (dry run)')
            else:
                self.get_logger().error(f'battery {v:.2f} V for {self.shutdown_s:.0f} s: powering off')
                subprocess.run(['sudo', '-n', 'systemctl', 'poweroff'], check=False)
            return
        if self.held('critical', v < self.critical, self.low_s):
            self.say('critical: charge now')
        elif self.held('low', v < self.low, self.low_s):
            self.say('low: charge soon')
        elif v >= self.low + 0.1 or (self.status == 'not connected' and v >= self.low):  # hysteresis
            self.say('ok')


def main():
    rclpy.init()
    node = BatteryGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
