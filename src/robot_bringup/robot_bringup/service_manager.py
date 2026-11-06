"""Starts and stops the robot's optional services, for the web UI's Services menu.

  in:  services/<id>/set_enabled  std_srvs/SetBool
  out: /services/status           latched JSON, {id: enabled}

Choices are saved in state_file, so a service switched off stays off after a restart. Only for
what the robot can drive without: the drive, its guard, odometry and the lidar stay in the launch
files. Cameras and stereo have their own switches (camera_manager.py, stereo_depth.py).
"""
import json
import os

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import String
from std_srvs.srv import SetBool

from robot_bringup.camera_manager import Proc

SERVICES = ('imu', 'sensors', 'crash', 'foxglove', 'rosbridge', 'claude')
DEFAULT_OFF = ('claude',)  # sends camera pictures off the robot: only when switched on


def launch_command(name):
    return ['ros2', 'launch', 'robot_bringup', 'optional.launch.py', f'service:={name}']


def load_choices(path):
    """Saved on/off per service; anything missing or unreadable is on (DEFAULT_OFF: off)."""
    try:
        with open(path) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    return {name: saved.get(name, name not in DEFAULT_OFF) is not False for name in SERVICES}


class ServiceManager(Node):
    def __init__(self, command=launch_command):
        super().__init__('service_manager')
        self.state_file = os.path.expanduser(self.declare_parameter('state_file', '~/.robot_services.json').value)
        self.enabled = load_choices(self.state_file)
        self.procs = {name: Proc(name, command(name), self.get_logger()) for name in SERVICES}
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/services/status', latched)
        for name in SERVICES:
            self.create_service(SetBool, f'services/{name}/set_enabled',
                                lambda req, res, name=name: self.on_set(name, req, res))
        self.publish_status()
        self.manage()
        self.create_timer(1.0, self.manage)

    def on_set(self, name, req, res):
        self.enabled[name] = req.data
        with open(self.state_file, 'w') as f:
            json.dump(self.enabled, f)
        self.manage()
        self.publish_status()
        res.success, res.message = True, f'{name} {"on" if req.data else "off"}'
        return res

    def publish_status(self):
        self.status_pub.publish(String(data=json.dumps(self.enabled)))

    def manage(self):
        for name, proc in self.procs.items():
            proc.update(self.enabled[name])

    def stop_all(self):
        for proc in self.procs.values():
            proc.stop_now()


def main():
    rclpy.init()
    node = ServiceManager()
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
