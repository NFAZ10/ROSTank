"""Start/stop mapping and navigation from Foxglove buttons (Service Call panel).

Services (std_srvs/Trigger, empty request):
  /robot/start_mapping     SLAM + Nav2 (robot_navigation navigation.launch.py)
  /robot/start_exploring   mapping + explorer.py driving to unexplored edges until the map is
                           complete (mode 'exploring'; back to 'mapping' when done, ready to save)
  /robot/save_map          save the SLAM map to ~/robot_ws/maps/<map_name param, else map_<time>>
                           and point latest.yaml at it
  /robot/start_navigation  Nav2 + AMCL on maps/<map_file param>.yaml (empty = the latest map)
  /robot/stop              stop whichever is running
Set the map_name / map_file parameters just before calling (the web UI does this).
Current mode is latched on /robot/nav_mode, the map being navigated on /robot/nav_map (std_msgs/String).

Each map keeps <name>.pose.json: where the robot was when the map was saved, then its last
AMCL pose while navigating. Navigation starts from that pose, so it is localized right away
if the robot hasn't been moved while off; otherwise set the pose once from the web UI.
"""
import json
import math
import os
import re
import signal
import subprocess
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

NAME_RE = re.compile(r'^[A-Za-z0-9_-]{1,40}$')


def yaw_of(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NavManager(Node):
    def __init__(self):
        super().__init__('nav_manager')
        self.map_dir = os.path.expanduser(
            self.declare_parameter('map_dir', '~/robot_ws/maps').value)
        os.makedirs(self.map_dir, exist_ok=True)
        self.latest = os.path.join(self.map_dir, 'latest.yaml')
        self.declare_parameter('map_name', '')   # next save_map
        self.declare_parameter('map_file', '')   # next start_navigation
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.nav_map = ''                       # name of the map being navigated
        self.last_pose, self.pose_saved_at = None, 0.0
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.on_amcl_pose, 10)

        self.proc = None
        self.mode = 'idle'
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.mode_pub = self.create_publisher(String, '/robot/nav_mode', latched)
        self.map_pub = self.create_publisher(String, '/robot/nav_map', latched)
        self.publish_mode()

        self.create_service(Trigger, '/robot/start_mapping', self.on_start_mapping)
        self.create_service(Trigger, '/robot/start_exploring', self.on_start_exploring)
        self.explorer = None
        self.create_service(Trigger, '/robot/save_map', self.on_save_map)
        self.create_service(Trigger, '/robot/start_navigation', self.on_start_navigation)
        self.create_service(Trigger, '/robot/stop', self.on_stop)
        self.create_timer(1.0, self.watch)

    def publish_mode(self):
        self.mode_pub.publish(String(data=self.mode))
        self.map_pub.publish(String(data=self.nav_map if self.mode == 'navigating' else ''))

    # ---------- per-map remembered pose ----------
    def pose_file(self, name):
        return os.path.join(self.map_dir, name + '.pose.json')

    def write_pose(self, name, x, y, yaw):
        with open(self.pose_file(name), 'w') as f:
            json.dump({'x': round(x, 3), 'y': round(y, 3), 'yaw': round(yaw, 4), 'time': time.time()}, f)

    def on_amcl_pose(self, msg):
        if self.mode == 'navigating':
            p = msg.pose.pose
            self.last_pose = (p.position.x, p.position.y, yaw_of(p.orientation))

    def store_last_pose(self):
        if self.mode == 'navigating' and self.nav_map and self.last_pose:
            self.write_pose(self.nav_map, *self.last_pose)
            self.pose_saved_at = time.monotonic()

    def latest_name(self):
        """Name of the map latest.yaml was copied from (its image: line)."""
        try:
            with open(self.latest) as f:
                for line in f:
                    if line.startswith('image:'):
                        return os.path.splitext(os.path.basename(line.split(':', 1)[1].strip()))[0]
        except OSError:
            pass
        return ''

    def launch(self, mode, extra_args):
        self.stop_proc()
        cmd = ['ros2', 'launch', 'robot_navigation', 'navigation.launch.py'] + extra_args
        # Own process group, so stop_proc can Ctrl-C the whole launch tree.
        self.proc = subprocess.Popen(cmd, start_new_session=True)
        self.mode = mode
        self.publish_mode()
        self.get_logger().info(f'{mode}: {" ".join(cmd)}')

    def stop_explorer(self):
        if self.explorer is not None and self.explorer.poll() is None:
            os.killpg(os.getpgid(self.explorer.pid), signal.SIGINT)
            try:
                self.explorer.wait(5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.explorer.pid), signal.SIGKILL)
        self.explorer = None

    def stop_proc(self):
        self.stop_explorer()
        if self.proc is None:
            return
        self.store_last_pose()
        if self.proc.poll() is None:
            pgid = os.getpgid(self.proc.pid)
            for sig, wait in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 2)):
                os.killpg(pgid, sig)
                try:
                    self.proc.wait(wait)
                    break
                except subprocess.TimeoutExpired:
                    continue
        self.proc = None
        self.mode = 'idle'
        self.last_pose = None
        self.publish_mode()

    def watch(self):
        # Keep the remembered pose fresh, so a power cut doesn't lose it.
        if self.last_pose and time.monotonic() - self.pose_saved_at > 5.0:
            self.store_last_pose()
        if self.explorer is not None and self.explorer.poll() is not None:
            code = self.explorer.returncode
            self.explorer = None
            if self.mode == 'exploring':
                self.mode = 'mapping'  # SLAM keeps running: the map can be saved
                self.publish_mode()
                self.get_logger().info('exploration finished' if code == 0 else f'explorer exited with {code}')
        if self.proc is not None and self.proc.poll() is not None:
            self.get_logger().warn(f'{self.mode} exited with code {self.proc.returncode}')
            self.proc = None
            self.mode = 'idle'
            self.publish_mode()

    def on_start_mapping(self, request, response):
        self.launch('mapping', [])
        response.success, response.message = True, 'mapping started: drive around, then Save map'
        return response

    def on_start_exploring(self, request, response):
        if self.mode not in ('mapping', 'exploring'):
            self.launch('mapping', [])
        self.stop_explorer()
        self.explorer = subprocess.Popen(['ros2', 'run', 'robot_bringup', 'explorer'], start_new_session=True)
        self.mode = 'exploring'
        self.publish_mode()
        self.get_logger().info('exploring')
        response.success = True
        response.message = 'exploring: the robot maps on its own until nothing is left, then comes back'
        return response

    def on_save_map(self, request, response):
        if self.mode not in ('mapping', 'exploring'):
            response.success, response.message = False, 'not mapping: start mapping first'
            return response
        name = self.get_parameter('map_name').value.strip() or time.strftime('map_%Y%m%d_%H%M%S')
        self.set_parameters([Parameter('map_name', value='')])
        if not NAME_RE.match(name) or name == 'latest':
            response.success = False
            response.message = 'map names: up to 40 letters, digits, - or _'
            return response
        path = os.path.join(self.map_dir, name)
        if os.path.exists(path + '.yaml'):
            response.success, response.message = False, f'a map called {name} already exists'
            return response
        result = subprocess.run(
            ['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', path,
             '--ros-args', '-p', 'save_map_timeout:=10.0'],
            capture_output=True, text=True, timeout=30)
        if result.returncode != 0 or not os.path.exists(path + '.yaml'):
            response.success = False
            response.message = 'save failed: ' + (result.stderr or result.stdout)[-300:]
            return response
        # latest.yaml = the new map's yaml with an absolute image path.
        with open(path + '.yaml') as f:
            lines = [f'image: {path}.pgm\n' if line.startswith('image:') else line for line in f]
        with open(self.latest, 'w') as f:
            f.writelines(lines)
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time()).transform
            self.write_pose(name, t.translation.x, t.translation.y, yaw_of(t.rotation))
        except Exception as e:  # map still usable, the pose just has to be set by hand
            self.get_logger().warn(f'no robot pose to store with {name}: {e}')
        response.success, response.message = True, f'saved map "{name}"'
        self.get_logger().info(response.message)
        return response

    def on_start_navigation(self, request, response):
        name = self.get_parameter('map_file').value.strip() or self.latest_name()
        self.set_parameters([Parameter('map_file', value='')])
        path = os.path.join(self.map_dir, name + '.yaml')
        if not name or not NAME_RE.match(name) or not os.path.exists(path):
            response.success = False
            response.message = f'no saved map called "{name}"' if name else 'no saved map yet: map and save first'
            return response
        args = [f'map:={path}']
        pose = None
        try:
            with open(self.pose_file(name)) as f:
                pose = json.load(f)
            args.append(f'initial_pose:={pose["x"]},{pose["y"]},{pose["yaw"]}')
        except (OSError, ValueError, KeyError):
            pass
        self.stop_proc()  # stores the previous session's pose under its own map first
        self.nav_map = name
        self.launch('navigating', args)
        response.success = True
        response.message = (f'navigating on "{name}" from where the robot last was. If it has moved, use Set pose.'
                            if pose else f'navigating on "{name}": use Set pose to show where the robot is')
        return response

    def on_stop(self, request, response):
        was = self.mode
        self.stop_proc()
        response.success, response.message = True, f'stopped ({was})'
        return response


def main():
    rclpy.init()
    node = NavManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_proc()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
