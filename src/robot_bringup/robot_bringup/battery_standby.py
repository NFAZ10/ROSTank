"""Keeps /battery going while the stack is stopped, so battery_guard can still protect the packs.

pico_bridge reads the battery while the stack runs. When it's down this node holds /dev/pico instead,
only reading: it never sends anything, so the motors stay stopped. It lets go as soon as the stack
starts; firmware/flash.sh stops this service while flashing.
"""
import os
import subprocess
import time

import rclpy
import serial
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import BatteryState

from robot_bringup.battery_filter import BatteryFilter

DEFAULT_DIVIDER = (6800 + 1500) / 1500


def parse_battery(line):
    """Pin volts from a Pico line like 'B 2392', or None."""
    parts = line.split()
    if len(parts) == 2 and parts[0] == 'B':
        try:
            return float(parts[1]) / 1000.0
        except ValueError:
            pass
    return None


def stack_is_down():
    state = subprocess.run(['systemctl', 'is-active', 'robot.service'], capture_output=True, text=True).stdout.strip()
    return state in ('inactive', 'failed')


class BatteryStandby(Node):
    def __init__(self):
        super().__init__('battery_standby')
        self.port = self.declare_parameter('port', '/dev/pico').value
        self.pub = self.create_publisher(BatteryState, 'battery', 10)
        self.filter = BatteryFilter()
        self.link, self.buffer = None, b''
        self.divider, self.scale = DEFAULT_DIVIDER, 1.0
        self.create_timer(1.0, self.manage)
        self.create_timer(0.1, self.read)

    def load_calibration(self):
        path = os.path.join(get_package_share_directory('robot_bringup'), 'config', 'drive.yaml')
        try:
            with open(path) as f:
                params = yaml.safe_load(f)['pico_bridge']['ros__parameters']
            self.divider = float(params.get('battery_divider', DEFAULT_DIVIDER))
            self.scale = float(params.get('battery_scale', 1.0))
        except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError) as e:
            self.get_logger().warning(f'using default battery calibration: {e}')

    def manage(self):
        if not stack_is_down():
            self.release()
        elif self.link is None and os.path.exists(self.port):
            try:
                self.link = serial.Serial(self.port, 115200, timeout=0)
                self.buffer = b''
                self.load_calibration()
                self.get_logger().info('stack down: reading the battery')
            except serial.SerialException as e:
                self.get_logger().warning(f'cannot open {self.port}: {e}', throttle_duration_sec=30.0)

    def release(self):
        if self.link is not None:
            self.link.close()
            self.link = None
            self.filter.hist.clear()
            self.get_logger().info('stack up: released the Pico')

    def read(self):
        if self.link is None:
            return
        try:
            self.buffer += self.link.read(self.link.in_waiting or 1)
        except (OSError, serial.SerialException):  # unplugged or being flashed
            self.link.close()
            self.link = None
            return
        *lines, self.buffer = self.buffer.split(b'\n')
        for line in lines:
            pin = parse_battery(line.decode(errors='ignore'))
            if pin is not None:
                msg = self.filter.message(time.monotonic(), pin, self.divider, self.scale)
                msg.header.stamp = self.get_clock().now().to_msg()
                self.pub.publish(msg)


def main():
    rclpy.init()
    node = BatteryStandby()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.release()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
