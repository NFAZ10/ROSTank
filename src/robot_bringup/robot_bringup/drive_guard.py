"""Drive guard: every drive command passes through here before the motors.

  /cmd_vel (Nav2, dock, calibration, robot move/turn)  ->  drive_guard  ->  /cmd_vel_safe  ->  pico_bridge
  /cmd_vel_manual (web UI joystick, robot drive)       ->

Looks along the direction of travel in a corridor the robot's width plus a margin: full
speed beyond slow_dist, slowing down closer in, stopped at stop_dist (turning on the spot
stays allowed, so the robot can be steered away). Obstacles come from the lidar (all
around) and, for manual driving only, stereo depth ahead (/depth/obstacles: things off the
lidar's plane, confirmed over several frames by stereo_depth.py). Not for Nav2: depth keeps both
cameras running, and with Nav2 too that overloads the CPU.

Sensors are only subscribed while driving and for wake_timeout after, so the lidar motor
(lidar_standby.py) and stereo depth (stereo_depth.py) still idle when nothing moves. While
the lidar spins up there is no fresh scan: until then only turning is let through.
Why it is holding back is latched on /drive_guard/status.

A crash (crash_monitor.py, /crash/event) stops motion that way for crash_hold seconds; backing away is fine.

Override, for driving out of a spot the guard won't allow: the drive_guard/override service
(SetBool) lets /cmd_vel_manual through unchecked, capped at override_speed. /cmd_vel stays
guarded, so Nav2 never gets it. It turns itself off override_timeout after the last manual
command.

/wheel/cmd (per-track bench test, tracks off) does not pass through here.
"""
import json
import math
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener


def matrix_of(t):
    """4x4 transform from a geometry_msgs Transform."""
    q, p = t.rotation, t.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    M = np.eye(4)
    M[:3, :3], M[:3, 3] = R, [p.x, p.y, p.z]
    return M


class DriveGuard(Node):
    def __init__(self):
        super().__init__('drive_guard')
        p = lambda name, default: self.declare_parameter(name, default).value
        self.half_length = p('half_length', 0.16)   # footprint (= Nav2 footprint), m
        self.half_width = p('half_width', 0.125)
        self.margin = p('side_margin', 0.04)       # corridor = footprint width + this each side
        self.stop_dist = p('stop_dist', 0.12)      # clearance at which forward/back motion stops
        self.slow_dist = p('slow_dist', 0.5)       # clearance where slowing down starts
        self.min_speed = p('min_speed', 0.05)      # slowed commands never go below this (m/s)
        self.wake_timeout = p('wake_timeout', 10.0)  # s after the last command before sensors idle
        self.use_depth = p('use_depth', True)
        self.override_speed = p('override_speed', 0.15)      # m/s limit while overridden
        self.override_timeout = p('override_timeout', 20.0)  # s after the last manual command

        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel_safe', 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/drive_guard/status', latched)
        self.create_subscription(Twist, 'cmd_vel', self.on_cmd, 10)
        self.create_subscription(Twist, 'cmd_vel_manual', lambda m: self.on_cmd(m, manual=True), 10)
        self.create_service(SetBool, 'drive_guard/override', self.on_override)
        self.create_subscription(String, '/crash/event', self.on_crash, 10)
        self.crash_hold = p('crash_hold', 1.5)
        self.crash_until, self.crash_direction = 0.0, 0
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.frames = {}  # sensor frame -> 4x4 into base_link (both mounts are fixed)

        self.cmd, self.cmd_time, self.sent_zero, self.cmd_manual = None, 0.0, True, False
        self.override_until = 0.0  # monotonic; 0 = not overridden
        self.subs, self.depth_sub, self.manual_time = None, None, 0.0
        self.scan_pts, self.scan_time = np.zeros((0, 2)), 0.0
        self.depth_pts, self.depth_time = np.zeros((0, 2)), 0.0
        self.status = None
        self.set_status('idle')
        self.create_timer(0.05, self.step)
        self.create_timer(1.0, self.manage)

    # ---------- sensors on demand ----------
    def wake(self):
        if self.subs is None:
            self.subs = [self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)]
        if self.use_depth and self.cmd_manual and self.depth_sub is None:
            self.depth_sub = self.create_subscription(PointCloud2, '/depth/obstacles', self.on_depth,
                                                      qos_profile_sensor_data)

    def manage(self):
        # Single-threaded executor: destroying here can't race a take on another thread.
        if self.depth_sub is not None and time.monotonic() - self.manual_time > self.wake_timeout:
            self.destroy_subscription(self.depth_sub)
            self.depth_sub = None
            self.depth_time = 0.0
        if self.subs is not None and time.monotonic() - self.cmd_time > self.wake_timeout:
            for sub in self.subs:
                self.destroy_subscription(sub)
            self.subs = None
            self.scan_time = self.depth_time = 0.0
            self.set_status('idle')

    def to_base(self, frame):
        if frame not in self.frames:
            try:
                self.frames[frame] = matrix_of(self.tf.lookup_transform('base_link', frame, Time()).transform)
            except Exception:  # noqa: BLE001 - TF not there yet; next message
                return None
        return self.frames[frame]

    def on_scan(self, msg):
        M = self.to_base(msg.header.frame_id)
        if M is None:
            return
        r = np.asarray(msg.ranges, np.float32)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment
        ok = np.isfinite(r) & (r > msg.range_min) & (r < msg.range_max)
        pts = np.stack([r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok]), np.zeros(ok.sum()), np.ones(ok.sum())])
        self.scan_pts, self.scan_time = (M @ pts)[:2].T, time.monotonic()

    def on_depth(self, msg):
        # Already in base_link and filtered; an empty cloud means nothing is there.
        xyz = np.frombuffer(msg.data, np.float32).reshape(-1, msg.point_step // 4)
        self.depth_pts, self.depth_time = xyz[:, :2].astype(float), time.monotonic()

    # ---------- guarding ----------
    def set_status(self, text):
        if self.override_until:
            text = f'override: protection OFF, {self.override_speed:.2f} m/s max'
        if text != self.status:
            self.status = text
            self.status_pub.publish(String(data=text))

    def on_override(self, req, res):
        self.override_until = time.monotonic() + self.override_timeout if req.data else 0.0
        self.get_logger().warning('override ON: manual driving unprotected' if req.data else 'override off')
        self.set_status('ready' if self.subs is not None else 'idle')
        res.success, res.message = True, self.status
        return res

    def on_crash(self, msg):
        self.crash_until = time.monotonic() + self.crash_hold
        self.crash_direction = json.loads(msg.data)['direction']

    def on_cmd(self, msg, manual=False):
        self.cmd, self.cmd_time, self.cmd_manual = msg, time.monotonic(), manual
        if manual:
            self.manual_time = self.cmd_time
            if self.override_until:
                self.override_until = self.cmd_time + self.override_timeout
        self.wake()
        self.step()  # react now, not on the next tick

    def clearance(self, direction):
        """Free distance from the front (direction 1) or back (-1) edge to the nearest obstacle."""
        now = time.monotonic()
        sets = [self.scan_pts]
        if direction > 0 and now - self.depth_time < 1.0:
            sets.append(self.depth_pts)  # cameras face forward
        pts = np.vstack(sets)
        x = pts[:, 0] * direction
        ahead = (np.abs(pts[:, 1]) < self.half_width + self.margin) & (x > self.half_length)
        return float(x[ahead].min() - self.half_length) if ahead.any() else math.inf

    def step(self):
        now = time.monotonic()
        if self.override_until and now > self.override_until:
            self.override_until = 0.0
            self.get_logger().info('override timed out')
            self.set_status('ready' if self.subs is not None else 'idle')
        if self.cmd is None or now - self.cmd_time > 0.5:
            if not self.sent_zero:
                self.cmd_pub.publish(Twist())
                self.sent_zero = True
            if self.subs is not None:
                self.set_status('ready')
            return
        out = Twist()
        out.angular.z = self.cmd.angular.z
        v = self.cmd.linear.x
        if self.override_until and self.cmd_manual:
            v = max(-self.override_speed, min(self.override_speed, v))
        elif now < self.crash_until and v * self.crash_direction > 0:
            v = 0.0
            self.set_status('stopped: crash detected')
        elif abs(v) > 1e-3:
            if now - self.scan_time > 0.5:
                v = 0.0
                self.set_status('waiting for lidar (turning only)')
            else:
                gap = self.clearance(1 if v > 0 else -1)
                where = 'ahead' if v > 0 else 'behind'
                if gap <= self.stop_dist:
                    v = 0.0
                    self.set_status(f'stopped: obstacle {max(gap, 0):.2f} m {where}')
                elif gap < self.slow_dist:
                    scale = (gap - self.stop_dist) / (self.slow_dist - self.stop_dist)
                    v = math.copysign(max(abs(v) * scale, min(abs(v), self.min_speed)), v)
                    self.set_status(f'slowing: obstacle {gap:.2f} m {where}')
                else:
                    self.set_status('clear')
        out.linear.x = v
        self.cmd_pub.publish(out)
        self.sent_zero = False


def main():
    rclpy.init()
    node = DriveGuard()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
