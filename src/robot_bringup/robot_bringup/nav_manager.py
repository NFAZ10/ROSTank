"""Start/stop mapping and navigation from Foxglove buttons (Service Call panel).

Services (std_srvs/Trigger, empty request):
  /robot/start_mapping     SLAM + Nav2 (robot_navigation navigation.launch.py)
  /robot/start_exploring   mapping + explorer.py driving to unexplored edges until the map is
                           complete (mode 'exploring'; back to 'mapping' when done, ready to save)
  /robot/stop_exploring    exploring -> mapping: the explorer stops, the map keeps going (auto-dock)
  /robot/save_map          save the SLAM map to ~/robot_ws/maps/<map_name param, else map_<time>>
                           and point latest.yaml at it; also slam_toolbox's pose graph
                           (<name>.posegraph/.data), so the map can be continued later
  /robot/continue_mapping  mapping that resumes maps/<map_file param> from its pose graph,
                           locating the robot near its remembered pose; save under the same
                           name to update that map
  /robot/start_navigation  Nav2 + AMCL on maps/<map_file param>.yaml (empty = the latest map)
  /robot/localize          while navigating: turn on the spot until AMCL is sure of the pose
                           (after Set pose), up to 2 turns
  /robot/find_myself       while navigating: AMCL guesses over the whole map, then turn until
                           it settles, up to 3 turns (no Set pose needed)
  /robot/stop              stop whichever is running
Set the map_name / map_file parameters just before calling (the web UI does this).
Current mode is latched on /robot/nav_mode, the map being navigated on /robot/nav_map (std_msgs/String),
localize progress on /robot/localize_status.

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
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import Empty, Trigger
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
        self.create_service(Trigger, '/robot/stop_exploring', self.on_stop_exploring)
        self.create_service(Trigger, '/robot/continue_mapping', self.on_continue_mapping)
        self.continue_name = ''  # map being continued (saving under this name may overwrite it)
        self.explorer = None
        self.create_service(Trigger, '/robot/save_map', self.on_save_map)
        self.create_service(Trigger, '/robot/start_navigation', self.on_start_navigation)
        self.create_service(Trigger, '/robot/stop', self.on_stop)
        self.create_timer(1.0, self.watch)

        # Localize / find myself: turn on the spot until AMCL's estimate is tight.
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.loc_pub = self.create_publisher(String, '/robot/localize_status', latched)
        self.global_loc = self.create_client(Empty, '/reinitialize_global_localization')
        self.create_service(Trigger, '/robot/localize', lambda req, res: self.on_localize(res, False))
        self.create_service(Trigger, '/robot/find_myself', lambda req, res: self.on_localize(res, True))
        self.loc = None          # running spin: {'start', 'turns', 'global'}
        self.amcl_cov = None     # (var x, var y, var yaw) from the latest /amcl_pose
        self.loc_pub.publish(String(data=''))
        self.create_timer(0.1, self.localize_step)

    def publish_mode(self):
        self.mode_pub.publish(String(data=self.mode))
        self.map_pub.publish(String(data=self.nav_map if self.mode == 'navigating'
                                    else self.continue_name if self.mode in ('mapping', 'exploring') else ''))

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
            c = msg.pose.covariance
            self.amcl_cov = (c[0], c[7], c[35])

    # ---------- localize / find myself ----------
    SPIN_RATE = 0.5           # rad/s: slow enough for a 7 Hz lidar to keep matching
    CONVERGED = (0.02, 0.02, 0.03)  # variances: about 14 cm and 10 deg standard deviation

    def on_localize(self, response, global_search):
        if self.mode != 'navigating':
            response.success, response.message = False, 'navigate on a saved map first'
            return response
        if global_search:
            if not self.global_loc.wait_for_service(timeout_sec=2.0):
                response.success, response.message = False, 'AMCL is not running yet: try again in a moment'
                return response
            self.global_loc.call_async(Empty.Request())  # AMCL spreads its guesses over the whole map
        self.amcl_cov = None
        self.loc = {'start': time.monotonic(), 'turns': 3 if global_search else 2, 'global': global_search}
        self.loc_pub.publish(String(data='finding myself on the map: turning…' if global_search
                                    else 'localizing: turning…'))
        response.success = True
        response.message = ('finding myself: the robot turns on the spot until it recognises the map'
                            if global_search else 'localizing: the robot turns on the spot until it is sure')
        return response

    def localize_step(self):
        if self.loc is None:
            return
        elapsed = time.monotonic() - self.loc['start']
        if self.mode != 'navigating':
            return self.localize_done('stopped')
        min_time = math.pi / self.SPIN_RATE  # at least half a turn, so the lidar sees all around
        cov = self.amcl_cov
        if elapsed > min_time and cov and all(v < lim for v, lim in zip(cov, self.CONVERGED)):
            return self.localize_done('localized: the lidar matches the map')
        if elapsed > self.loc['turns'] * 2 * math.pi / self.SPIN_RATE:
            return self.localize_done("not sure where I am: use Set pose near the robot's real spot, then Localize")
        t = Twist()
        t.angular.z = self.SPIN_RATE
        self.cmd_pub.publish(t)

    def localize_done(self, text):
        self.loc = None
        for _ in range(3):
            self.cmd_pub.publish(Twist())
        self.loc_pub.publish(String(data=text))
        self.get_logger().info(text)
        self.store_last_pose()

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
        if self.get_parameter('use_sim_time').value:  # testing against the Gazebo sim
            cmd.append('use_sim_time:=true')
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
        self.continue_name = ''
        if self.loc is not None:
            self.localize_done('stopped')
        self.loc_pub.publish(String(data=''))
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
        self.clear_pending_dock()
        self.launch('mapping', [])
        response.success, response.message = True, 'mapping started: drive around, then Save map'
        return response

    def clear_pending_dock(self):
        for name in ('.pending_dock.json', '.pending_crash.json'):  # belong to the previous mapping run
            pending = os.path.join(self.map_dir, name)
            if os.path.exists(pending):
                os.remove(pending)

    def on_stop_exploring(self, request, response):
        if self.mode != 'exploring':
            response.success, response.message = False, 'not exploring'
            return response
        self.stop_explorer()
        self.mode = 'mapping'
        self.publish_mode()
        self.get_logger().info('exploring stopped, still mapping')
        response.success, response.message = True, 'stopped exploring (still mapping)'
        return response

    def on_start_exploring(self, request, response):
        if self.mode not in ('mapping', 'exploring'):
            self.clear_pending_dock()
            self.launch('mapping', [])
        self.stop_explorer()
        cmd = ['ros2', 'run', 'robot_bringup', 'explorer']
        if self.get_parameter('use_sim_time').value:
            cmd += ['--ros-args', '-p', 'use_sim_time:=true']
        self.explorer = subprocess.Popen(cmd, start_new_session=True)
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
        if os.path.exists(path + '.yaml') and name != self.continue_name:
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
        # slam_toolbox's own pose graph, so this map can be continued later (best effort). The
        # `ros2 service call` CLI can hang after slam_toolbox has already written the files, so
        # don't wait on it: start it, and wait for the files instead.
        for ext in ('.posegraph', '.data'):
            if os.path.exists(path + ext):
                os.remove(path + ext)  # continuing: don't mistake the old graph for the new one
        cli = subprocess.Popen(
            ['ros2', 'service', 'call', '/slam_toolbox/serialize_map', 'slam_toolbox/srv/SerializePoseGraph',
             f'{{filename: "{path}"}}'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        end = time.monotonic() + 30.0
        while time.monotonic() < end and not (os.path.exists(path + '.posegraph') and os.path.exists(path + '.data')):
            time.sleep(0.5)
        time.sleep(1.0)  # let slam_toolbox finish writing
        if cli.poll() is None:
            os.killpg(os.getpgid(cli.pid), signal.SIGKILL)
        if not os.path.exists(path + '.posegraph'):
            self.get_logger().warn(f'pose graph not saved for {name}: it can be navigated but not continued')
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
        pending = os.path.join(self.map_dir, '.pending_dock.json')  # dock seen while mapping (dock.py)
        if os.path.exists(pending):
            os.replace(pending, os.path.join(self.map_dir, name + '.dock.json'))
        pending = os.path.join(self.map_dir, '.pending_crash.json')
        if os.path.exists(pending):
            os.replace(pending, os.path.join(self.map_dir, name + '.crash.json'))
        response.success, response.message = True, f'saved map "{name}"' + (' with its dock' if os.path.exists(
            os.path.join(self.map_dir, name + '.dock.json')) else '')
        self.get_logger().info(response.message)
        return response

    def on_continue_mapping(self, request, response):
        name = self.get_parameter('map_file').value.strip()
        self.set_parameters([Parameter('map_file', value='')])
        path = os.path.join(self.map_dir, name)
        if not NAME_RE.match(name or '-') or not os.path.exists(path + '.posegraph'):
            response.success = False
            response.message = (f'"{name}" was saved before maps could be continued: map it again'
                                if os.path.exists(path + '.yaml') else f'no saved map called "{name}"')
            return response
        x = y = yaw = 0.0
        try:
            with open(self.pose_file(name)) as f:
                pose = json.load(f)
            x, y, yaw = pose['x'], pose['y'], pose['yaw']
        except (OSError, ValueError, KeyError):
            pass
        self.clear_pending_dock()
        self.launch('mapping', [])
        self.continue_name = name
        self.publish_mode()
        # Waits for slam_toolbox to come up, then loads the graph with the robot at (x, y, yaw)
        # (START_AT_GIVEN_POSE = 2: the only kind slam_toolbox allows in mapping mode; scan
        # matching then refines the pose as it drives). Runs on its own; mapping carries on.
        subprocess.Popen(
            ['timeout', '120', 'ros2', 'service', 'call', '/slam_toolbox/deserialize_map',
             'slam_toolbox/srv/DeserializePoseGraph',
             f'{{filename: "{path}", match_type: 2, initial_pose: {{x: {x}, y: {y}, theta: {yaw}}}}}'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        response.success = True
        response.message = (f'continuing "{name}": start with the robot near where it last was on that map. '
                            'Save with the same name to update it.')
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
