"""Calibrated IMU from the sensor Pico's MPU6050-type chip (raw data comes through pico_io.py).

  in:  /imu/raw       sensor_msgs/Imu, frame mpu_imu (the chip's printed axes)
       /cmd_vel_safe  to know when the robot is meant to be still
  out: /imu/data      same, gyro bias removed, no orientation
       /imu/status    latched text: why there is or isn't data

This chip's gyro reads off by up to 11 deg/s at rest and the offset creeps as it warms up, so nothing
is published until a first bias is measured, and it keeps being re-measured whenever the robot rests.
"""
from collections import deque
import statistics
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String

GYRO_VAR = 0.021 ** 2   # (rad/s)^2, kept from the APM so the EKF weighs the gyro the same
ACCEL_VAR = 0.046 ** 2  # (m/s^2)^2
SETTLE_S = 1.0          # after the last motion command


class GyroBias:
    """Mean gyro over quiet windows. Each window is judged whole, so a nudge spoils only that window."""

    def __init__(self, samples=100, still_sd=0.01, blend=0.1):
        self.samples, self.still_sd, self.blend = samples, still_sd, blend  # 2 s at 50 Hz; 0.01 rad/s = 0.6 deg/s
        self.window = deque(maxlen=samples)
        self.bias = None

    def add(self, gyro, moving):
        if moving:
            self.window.clear()
            return
        self.window.append(gyro)
        if len(self.window) < self.samples:
            return
        axes = list(zip(*self.window))
        self.window.clear()
        if max(statistics.pstdev(a) for a in axes) > self.still_sd:
            return
        mean = tuple(statistics.fmean(a) for a in axes)
        if self.bias is None:
            self.bias = mean
        else:
            self.bias = tuple(b + self.blend * (m - b) for b, m in zip(self.bias, mean))

    def correct(self, gyro):
        return tuple(g - b for g, b in zip(gyro, self.bias))


class MpuImu(Node):
    def __init__(self):
        super().__init__('mpu_imu')
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/imu/status', latched)
        self.imu_pub = self.create_publisher(Imu, '/imu/data', qos_profile_sensor_data)
        self.bias = GyroBias()
        self.status, self.last_raw, self.last_motion = None, -1e9, -1e9
        self.set_status('no data from the sensor board')
        self.create_subscription(Imu, '/imu/raw', self.on_raw, qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_vel_safe', self.on_cmd, 10)
        self.create_timer(1.0, self.check_stale)

    def set_status(self, text):
        if text != self.status:
            self.status = text
            self.status_pub.publish(String(data=text))
            self.get_logger().info(text)

    def on_cmd(self, msg):
        if abs(msg.linear.x) > 1e-3 or abs(msg.angular.z) > 1e-3:
            self.last_motion = time.monotonic()

    def check_stale(self):
        if time.monotonic() - self.last_raw > 1.0:
            self.set_status('no data from the sensor board')

    def on_raw(self, msg):
        now = time.monotonic()
        self.last_raw = now
        g = msg.angular_velocity
        gyro = (g.x, g.y, g.z)
        self.bias.add(gyro, now - self.last_motion < SETTLE_S)
        if self.bias.bias is None:
            self.set_status('calibrating: keep the robot still')
            return
        self.set_status('ok')
        g.x, g.y, g.z = self.bias.correct(gyro)
        msg.orientation_covariance[0] = -1.0  # no orientation (REP 145)
        for i in (0, 4, 8):
            msg.angular_velocity_covariance[i] = GYRO_VAR
            msg.linear_acceleration_covariance[i] = ACCEL_VAR
        self.imu_pub.publish(msg)


def main():
    rclpy.init()
    node = MpuImu()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
