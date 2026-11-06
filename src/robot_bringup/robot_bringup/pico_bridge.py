"""Bridge between ROS 2 and the Pico drive firmware (see firmware/pico_drive).

Subscribes cmd_vel (remapped to /cmd_vel_safe: drive_guard's output), sends per-track sprocket speeds to the Pico, and turns the
Pico's cumulative sprocket angles into /odom (+ optional odom->base_link TF).

Per-track view for testing (web UI encoder page):
  wheel/joint_states  sensor_msgs/JointState [left, right]: angle [rad], measured and
                      commanded sprocket speed [rad/s] (velocity, effort)
  wheel/cmd           std_msgs/Float64MultiArray [left, right] sprocket speeds [rad/s];
                      like /cmd_vel, the newest command of either kind wins

Battery: the Pico reports the divider voltage at GP26 ("B <mV>"); published as /battery
(sensor_msgs/BatteryState, 10 s average, LiFePO4 percentage; present=False if not wired).
"""
import collections
import math
import threading
import time

import rclpy
import serial
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from sensor_msgs.msg import BatteryState, JointState
from std_msgs.msg import Float64MultiArray
from tf2_ros import TransformBroadcaster

from robot_bringup.battery_filter import BatteryFilter


class PicoBridge(Node):
    def __init__(self):
        super().__init__('pico_bridge')
        self.port = self.declare_parameter('port', '/dev/pico').value
        self.wheel_radius = self.declare_parameter('wheel_radius', 0.03).value
        # Effective track width: tracks skid when turning, so calibrate this by
        # commanding a 360 deg turn and scaling until odom agrees.
        self.track_width = self.declare_parameter('track_width', 0.20).value
        self.max_wheel_speed = self.declare_parameter('max_wheel_speed', 13.0).value
        self.cmd_timeout = self.declare_parameter('cmd_timeout', 0.5).value
        self.publish_tf = self.declare_parameter('publish_tf', True).value
        self.odom_frame = self.declare_parameter('odom_frame', 'odom').value
        self.base_frame = self.declare_parameter('base_frame', 'base_link').value
        # Wiring fixes without reflashing: flip a track's direction, or swap the sides.
        self.invert_left = self.declare_parameter('invert_left', False).value
        self.invert_right = self.declare_parameter('invert_right', False).value
        self.swap_sides = self.declare_parameter('swap_sides', False).value
        # Battery divider: 6.8k over 1.5k by default. battery_scale trims it against a multimeter.
        self.battery_divider = self.declare_parameter('battery_divider', (6800 + 1500) / 1500).value
        self.battery_scale = self.declare_parameter('battery_scale', 1.0).value
        self.battery_filter = BatteryFilter()
        self.battery_pub = self.create_publisher(BatteryState, 'battery', 10)
        # Calibration (web UI) changes these live.
        self.add_on_set_parameters_callback(self.on_params)

        self.odom_pub = self.create_publisher(Odometry, 'odom', 10)
        self.tf_pub = TransformBroadcaster(self) if self.publish_tf else None
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd_vel, 10)
        self.joint_pub = self.create_publisher(JointState, 'wheel/joint_states', 10)
        self.create_subscription(Float64MultiArray, 'wheel/cmd', self.on_wheel_cmd, 10)
        # Speed over ~0.1 s: one hall edge per 20 ms sample alone is 0.4 rad/s of noise.
        self.angle_hist = collections.deque(maxlen=6)
        self.create_timer(0.05, self.send_cmd)

        self.ser = None
        self.ser_lock = threading.Lock()
        self.cmd = (0.0, 0.0)
        self.last_cmd_time = 0.0
        self.x = self.y = self.yaw = 0.0
        self.last_angles = None
        self.last_odom_time = None

        threading.Thread(target=self.reader, daemon=True).start()

    LIVE_PARAMS = ('wheel_radius', 'track_width', 'invert_left', 'invert_right', 'swap_sides',
                   'battery_divider', 'battery_scale')

    def on_params(self, params):
        for p in params:
            if p.name in ('wheel_radius', 'track_width', 'battery_divider', 'battery_scale') and p.value <= 0:
                return SetParametersResult(successful=False, reason=f'{p.name} must be > 0')
        for p in params:
            if p.name in self.LIVE_PARAMS:
                setattr(self, p.name, p.value)
        return SetParametersResult(successful=True)

    def to_pico(self, left, right):
        """Robot left/right track values -> the Pico's channels (and back: it's symmetric)."""
        if self.invert_left:
            left = -left
        if self.invert_right:
            right = -right
        return (right, left) if self.swap_sides else (left, right)

    def from_pico(self, a, b):
        if self.swap_sides:
            a, b = b, a
        return (-a if self.invert_left else a, -b if self.invert_right else b)

    def on_cmd_vel(self, msg):
        v, w = msg.linear.x, msg.angular.z
        left = (v - w * self.track_width / 2.0) / self.wheel_radius
        right = (v + w * self.track_width / 2.0) / self.wheel_radius
        # Scale both sides together so the turn radius is kept when saturating.
        peak = max(abs(left), abs(right))
        if peak > self.max_wheel_speed:
            left *= self.max_wheel_speed / peak
            right *= self.max_wheel_speed / peak
        self.cmd = self.to_pico(left, right)
        self.last_cmd_time = time.monotonic()

    def on_wheel_cmd(self, msg):
        if len(msg.data) != 2:
            return
        clip = lambda w: max(-self.max_wheel_speed, min(self.max_wheel_speed, w))
        self.cmd = self.to_pico(clip(msg.data[0]), clip(msg.data[1]))
        self.last_cmd_time = time.monotonic()

    def send_cmd(self):
        if time.monotonic() - self.last_cmd_time > self.cmd_timeout:
            self.cmd = (0.0, 0.0)
        self.write(f'V {self.cmd[0]:.4f} {self.cmd[1]:.4f}\n')

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
        self.last_angles = None
        self.angle_hist.clear()

    def reader(self):
        while rclpy.ok():
            if self.ser is None:
                try:
                    s = serial.Serial(self.port, 115200, timeout=0.2)
                    with self.ser_lock:
                        self.ser = s
                    self.get_logger().info(f'connected to {self.port}')
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
            parts = raw.decode(errors='ignore').split()
            if len(parts) == 3 and parts[0] == 'E':
                try:
                    self.on_angles(*self.from_pico(float(parts[1]), float(parts[2])))
                except ValueError:
                    pass
            elif len(parts) == 2 and parts[0] == 'B':
                try:
                    self.on_battery(float(parts[1]) / 1000.0)
                except ValueError:
                    pass

    def on_battery(self, pin_volts):
        msg = self.battery_filter.message(time.monotonic(), pin_volts, self.battery_divider, self.battery_scale)
        msg.header.stamp = self.get_clock().now().to_msg()
        self.battery_pub.publish(msg)

    def publish_joints(self, now, left, right):
        t = now.nanoseconds * 1e-9
        self.angle_hist.append((t, left, right))
        t0, l0, r0 = self.angle_hist[0]
        dt = t - t0
        cmd = self.from_pico(*self.cmd)
        js = JointState()
        js.header.stamp = now.to_msg()
        js.name = ['left_sprocket', 'right_sprocket']
        js.position = [left, right]
        js.velocity = [(left - l0) / dt, (right - r0) / dt] if dt > 0 else [0.0, 0.0]
        js.effort = [float(cmd[0]), float(cmd[1])]  # commanded speed, not a torque
        self.joint_pub.publish(js)

    def on_angles(self, left, right):
        now = self.get_clock().now()
        self.publish_joints(now, left, right)
        if self.last_angles is None:
            self.last_angles, self.last_odom_time = (left, right), now
            return
        dl = (left - self.last_angles[0]) * self.wheel_radius
        dr = (right - self.last_angles[1]) * self.wheel_radius
        dt = (now - self.last_odom_time).nanoseconds * 1e-9
        self.last_angles, self.last_odom_time = (left, right), now

        ds = (dl + dr) / 2.0
        dyaw = (dr - dl) / self.track_width
        self.x += ds * math.cos(self.yaw + dyaw / 2.0)
        self.y += ds * math.sin(self.yaw + dyaw / 2.0)
        self.yaw += dyaw

        qz, qw = math.sin(self.yaw / 2.0), math.cos(self.yaw / 2.0)
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        if dt > 0:
            odom.twist.twist.linear.x = ds / dt
            odom.twist.twist.angular.z = dyaw / dt
        # Tracks slip, yaw especially: trust VSLAM/lidar over this when fusing.
        odom.pose.covariance[0] = odom.pose.covariance[7] = 0.05
        odom.pose.covariance[35] = 0.2
        odom.twist.covariance[0] = 0.05
        odom.twist.covariance[35] = 0.2
        self.odom_pub.publish(odom)

        if self.tf_pub:
            t = TransformStamped()
            t.header = odom.header
            t.child_frame_id = self.base_frame
            t.transform.translation.x = self.x
            t.transform.translation.y = self.y
            t.transform.rotation.z = qz
            t.transform.rotation.w = qw
            self.tf_pub.sendTransform(t)

    def stop(self):
        self.write('S\n')


def main():
    rclpy.init()
    node = PicoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
