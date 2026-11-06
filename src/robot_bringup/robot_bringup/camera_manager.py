"""Runs each camera only while something wants it (the cameras are the stack's biggest CPU load).

Watches who subscribes to each side's topics, so nothing has to ask for a camera: watching the
left view runs the left camera, stereo_depth runs both, nobody leaves both off. The JPEG encoder
(CPU, needed by viewers but not by depth) is a process of its own, started the same way.

A camera takes about 20 s to come up (Argus open + NITROS negotiation, see cameras.launch.py),
so it stays on for `linger` seconds after the last user. State is latched on /cameras/status.
cameras/set_enabled (web UI Services menu) turns them all off whoever is asking; that survives restarts.
"""
import os
import signal
import subprocess
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import SetBool

SIDES = ('left', 'right')
# Images only: viewers like Foxglove subscribe to camera_info by themselves, which needs no camera.
CAMERA_TOPICS = ('image_raw', 'preview/image')
JPEG_TOPIC = 'preview/image/compressed'


class Proc:
    """A process group: started when wanted, stopped with SIGINT, then SIGKILL if it hangs."""

    def __init__(self, name, cmd, logger):
        self.name, self.cmd, self.log = name, cmd, logger
        self.proc, self.kill_at, self.retry_at = None, None, 0.0

    def signal(self, sig):
        try:
            os.killpg(self.proc.pid, sig)
        except ProcessLookupError:
            pass

    def update(self, wanted):
        now = time.monotonic()
        if self.proc is not None and self.proc.poll() is not None:
            if self.kill_at is None:
                self.log.warning(f'{self.name} exited by itself ({self.proc.returncode})')
                self.retry_at = now + 5.0
            self.proc, self.kill_at = None, None
        if self.proc is None:
            if wanted and now >= self.retry_at:
                self.log.info(f'{self.name} on')
                self.proc = subprocess.Popen(self.cmd, start_new_session=True)
        elif self.kill_at is not None:
            if now > self.kill_at:
                self.signal(signal.SIGKILL)
        elif not wanted:
            self.log.info(f'{self.name} off')
            self.signal(signal.SIGINT)
            self.kill_at = now + 10.0

    @property
    def running(self):
        return self.proc is not None and self.kill_at is None

    def stop_now(self):
        if self.proc is None:
            return
        self.signal(signal.SIGINT)
        try:
            self.proc.wait(5.0)
        except subprocess.TimeoutExpired:
            self.signal(signal.SIGKILL)


class CameraManager(Node):
    def __init__(self):
        super().__init__('camera_manager')
        self.linger = self.declare_parameter('linger', 30.0).value
        self.disabled_file = os.path.expanduser(self.declare_parameter('disabled_file', '~/.robot_cameras_disabled').value)
        self.enabled = not os.path.exists(self.disabled_file)
        self.create_service(SetBool, 'cameras/set_enabled', self.on_set_enabled)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/cameras/status', latched)
        self.status = None
        log = self.get_logger()
        self.cameras = {side: Proc(f'{side} camera', [
            'ros2', 'launch', 'robot_bringup', 'cameras.launch.py', f'side:={side}', 'preview:=false'], log)
            for side in SIDES}
        self.jpegs = {side: Proc(f'{side} jpeg', [
            'ros2', 'run', 'image_transport', 'republish', 'raw', 'compressed', '--ros-args',
            '-r', f'__node:=preview_jpeg_{side}', '-p', 'out.jpeg_quality:=60',  # ~1/3 the size of the default 95
            '-r', f'in:={side}/preview/image', '-r', f'out/compressed:={side}/{JPEG_TOPIC}'], log)
            for side in SIDES}
        self.last_camera = {side: 0.0 for side in SIDES}
        self.last_jpeg = {side: 0.0 for side in SIDES}
        self.create_timer(1.0, self.manage)

    def on_set_enabled(self, req, res):
        self.enabled = req.data
        if self.enabled:
            if os.path.exists(self.disabled_file):
                os.remove(self.disabled_file)
        else:
            open(self.disabled_file, 'w').close()
            self.last_camera = {side: 0.0 for side in SIDES}
            self.last_jpeg = {side: 0.0 for side in SIDES}
        self.manage()
        res.success, res.message = True, 'cameras on' if self.enabled else 'cameras off'
        return res

    def used(self, side, topics):
        own = (f'preview_resize_{side}', f'preview_jpeg_{side}', f'argus_{side}')
        return any(info.node_name not in own for topic in topics
                   for info in self.get_subscriptions_info_by_topic(f'/{side}/{topic}'))

    def manage(self):
        now = time.monotonic()
        for side in SIDES:
            if not self.enabled:
                pass
            elif self.used(side, (JPEG_TOPIC,)):
                self.last_jpeg[side] = self.last_camera[side] = now
            elif self.used(side, CAMERA_TOPICS):
                self.last_camera[side] = now
            self.cameras[side].update(now - self.last_camera[side] < self.linger and self.last_camera[side] > 0)
            self.jpegs[side].update(now - self.last_jpeg[side] < self.linger and self.last_jpeg[side] > 0)
        text = 'disabled' if not self.enabled else ', '.join(
            f'{side} ' + ('off' if not self.cameras[side].running else
                          'on + jpeg' if self.jpegs[side].running else 'on') for side in SIDES)
        if text != self.status:
            self.status = text
            self.status_pub.publish(String(data=text))

    def stop_all(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # launch may interrupt again while we wait for these
        for proc in (*self.jpegs.values(), *self.cameras.values()):
            proc.stop_now()


def main():
    rclpy.init()
    node = CameraManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.stop_all()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
