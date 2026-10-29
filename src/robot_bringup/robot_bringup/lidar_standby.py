"""Spin the RPLIDAR only while something consumes /scan.

rplidar_ros' own auto_standby never stops an A1 that has not scanned yet (its
motor runs from the moment the serial port opens), so this drives the
start_motor/stop_motor services from the observed state instead:
  - scans arriving but no other subscribers  -> stop_motor
  - other subscribers but no scans arriving   -> start_motor
Being state-free, it also recovers when rplidar_node respawns.
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty


class LidarStandby(Node):
    def __init__(self):
        super().__init__('lidar_standby')
        self.scan_topic = self.declare_parameter('scan_topic', '/scan').value
        # Keep spinning this long after the last consumer leaves, so brief
        # reconnects (e.g. Foxglove layout changes) don't cycle the motor.
        self.idle_timeout = self.declare_parameter('idle_timeout', 5.0).value

        self.create_subscription(LaserScan, self.scan_topic, self.on_scan, qos_profile_sensor_data)
        self.start_cli = self.create_client(Empty, 'start_motor')
        self.stop_cli = self.create_client(Empty, 'stop_motor')
        self.last_scan = 0.0
        self.last_consumer = 0.0
        self.spinup_until = 0.0
        self.quiet_until = 0.0
        self.pending = None
        self.create_timer(0.5, self.tick)

    def on_scan(self, _msg):
        self.last_scan = time.monotonic()

    def tick(self):
        now = time.monotonic()
        consumers = self.count_subscribers(self.scan_topic) - 1  # minus ourselves
        if consumers > 0:
            self.last_consumer = now
        # Scans already in flight keep arriving briefly after a stop; ignore them.
        scanning = (now - self.last_scan < 1.5 and now > self.quiet_until) or now < self.spinup_until
        if self.pending is not None and not self.pending.done():
            return
        if consumers > 0 and not scanning:
            self.call(self.start_cli, 'start')
        elif consumers == 0 and scanning and now - self.last_consumer > self.idle_timeout:
            self.call(self.stop_cli, 'stop')

    def call(self, client, what):
        if not client.service_is_ready():
            return
        self.get_logger().info(f'{what} lidar motor')
        self.pending = client.call_async(Empty.Request())
        if what == 'start':
            # Spin-up takes ~2 s; don't re-request while it gets going.
            self.spinup_until = time.monotonic() + 4.0
        else:
            self.quiet_until = time.monotonic() + 2.0


def main():
    rclpy.init()
    node = LidarStandby()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
