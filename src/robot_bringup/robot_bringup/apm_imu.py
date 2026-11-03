"""APM flight controller (Mega 2560, ArduPlane) as an IMU: MAVLink RAW_IMU -> sensor_msgs/Imu.

  out: /imu/apm      sensor_msgs/Imu, frame apm_imu (x forward, y left, z up on the board), no orientation
       /apm/status   latched text: why there is or isn't data

Only reads: it asks for the raw sensor stream and never arms or commands anything. The board's
compass and GPS aren't fitted and its barometer reads wrong (514 hPa, 2026-11-03), so gyro and
accelerometer are all this publishes.

Opening the port resets the board (DTR), and at boot it sweeps its servo outputs and calibrates
its gyros, which needs it still for ~15 s. Clearing HUPCL makes later opens not reset it; data is
only published once it reports it has finished booting.

Keep the board's outputs disconnected: this firmware drives them even when disarmed.
"""
import os
import termios
import time

import rclpy
from pymavlink import mavutil
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import Imu
from std_msgs.msg import String

G = 9.80665
GYRO_VAR = 0.021 ** 2   # (rad/s)^2, worst axis measured still, 2026-11-03
ACCEL_VAR = 0.046 ** 2  # (m/s^2)^2


def to_flu(x, y, z):
    """ArduPilot's body axes (forward, right, down) to ROS's (forward, left, up)."""
    return x, -y, -z


def imu_values(raw):
    """RAW_IMU (milli-g, milli-rad/s) -> gyro rad/s, accel m/s^2, both forward-left-up."""
    gyro = to_flu(raw.xgyro / 1000.0, raw.ygyro / 1000.0, raw.zgyro / 1000.0)
    accel = to_flu(raw.xacc / 1000.0 * G, raw.yacc / 1000.0 * G, raw.zacc / 1000.0 * G)
    return gyro, accel


def booted(heartbeat):
    return heartbeat.system_status not in (mavutil.mavlink.MAV_STATE_UNINIT, mavutil.mavlink.MAV_STATE_BOOT,
                                           mavutil.mavlink.MAV_STATE_CALIBRATING)


class ApmImu(Node):
    def __init__(self):
        super().__init__('apm_imu')
        self.port = self.declare_parameter('port', '/dev/apm').value
        self.rate = self.declare_parameter('rate', 50).value
        self.frame = self.declare_parameter('frame', 'apm_imu').value
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/apm/status', latched)
        self.imu_pub = self.create_publisher(Imu, '/imu/apm', qos_profile_sensor_data)
        self.link, self.status = None, None
        self.ready, self.stream_sent, self.last_imu = False, 0.0, 0.0
        self.count, self.count_since = 0, time.monotonic()
        self.set_status('starting')

    def set_status(self, text):
        if text != self.status:
            self.status = text
            self.status_pub.publish(String(data=text))
            self.get_logger().info(text)

    def open(self):
        try:
            self.link = mavutil.mavlink_connection(self.port, baud=115200, autoreconnect=False)
            attrs = termios.tcgetattr(self.link.fd)
            attrs[2] &= ~termios.HUPCL
            termios.tcsetattr(self.link.fd, termios.TCSANOW, attrs)
        except (OSError, termios.error) as e:
            self.link = None
            self.set_status(f'not connected ({os.path.basename(self.port)}: {e.strerror or e})')
            return False
        self.ready, self.stream_sent, self.last_imu = False, 0.0, time.monotonic()
        self.set_status('booting: keep the robot still')
        return True

    def close(self):
        if self.link is not None:
            self.link.close()
        self.link = None

    def on_message(self, msg):
        kind = msg.get_type()
        now = time.monotonic()
        if kind == 'HEARTBEAT' and msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
            self.ready = booted(msg)
            if self.ready and now - self.stream_sent > 5.0 and now - self.last_imu > 1.0:
                # Asked again if it goes quiet: the board forgets after a reset (or saves it, either is fine).
                self.link.mav.request_data_stream_send(
                    msg.get_srcSystem(), msg.get_srcComponent(), mavutil.mavlink.MAV_DATA_STREAM_RAW_SENSORS,
                    self.rate, 1)
                self.stream_sent = now
        elif kind == 'RAW_IMU' and self.ready:
            self.last_imu = now
            self.publish_imu(msg)

    def publish_imu(self, raw):
        gyro, accel = imu_values(raw)
        m = Imu()
        m.header.stamp, m.header.frame_id = self.get_clock().now().to_msg(), self.frame
        m.orientation_covariance[0] = -1.0  # no orientation (REP 145)
        m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z = gyro
        m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z = accel
        for i in (0, 4, 8):
            m.angular_velocity_covariance[i] = GYRO_VAR
            m.linear_acceleration_covariance[i] = ACCEL_VAR
        self.imu_pub.publish(m)
        self.count += 1

    def report(self):
        now = time.monotonic()
        if now - self.count_since >= 5.0:
            hz = self.count / (now - self.count_since)
            self.count, self.count_since = 0, now
            if hz > 1:
                self.set_status('ok')
            elif self.ready:
                self.set_status('connected, no IMU data')

    def run(self):
        while rclpy.ok():
            if self.link is None and not self.open():
                time.sleep(5.0)
                continue
            try:
                msg = self.link.recv_match(blocking=True, timeout=0.2)
            except (OSError, termios.error) as e:  # unplugged
                self.close()
                self.set_status(f'disconnected ({e})')
                continue
            if msg is not None:
                self.on_message(msg)
            self.report()


def main():
    rclpy.init()
    node = ApmImu()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
