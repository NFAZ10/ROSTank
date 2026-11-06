"""Full robot: Pico drive + lidar, stereo cameras, Foxglove bridge for remote viewing.

Connect Foxglove (app or https://app.foxglove.dev) to ws://<jetson-ip>:8765.
rosbridge (roslibjs pages, phone apps) on ws://<jetson-ip>:9090.
Mapping/navigation start and stop through the /robot/* services (nav_manager.py).
Web UI on http://<jetson-ip>:8080 runs separately (robot-web.service), so it can stop/start this.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node



def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory('robot_bringup'), 'launch')
    description_dir = os.path.join(get_package_share_directory('robot_description'), 'launch')

    def include(name, directory=launch_dir):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(directory, name)))

    return LaunchDescription([
        include('drive.launch.py'),
        # Cameras run only while something subscribes to them.
        Node(package='robot_bringup', executable='camera_manager', output='screen'),
        # Sensor mounts come from the URDF (robot_description/urdf/robot.urdf.xacro).
        include('description.launch.py', description_dir),
        # Stereo depth from the camera previews, only while something watches it.
        Node(package='robot_bringup', executable='stereo_depth', output='screen'),
        # Home dock by AprilTag (/robot/go_dock); idle and camera-free until called.
        Node(package='robot_bringup', executable='dock', output='screen'),
        # Mapping / navigation on demand (Foxglove Service Call buttons on /robot/*).
        Node(package='robot_bringup', executable='nav_manager', output='screen'),
        # Switchable from the web UI: IMU, sensor/LED board, Foxglove, rosbridge.
        Node(package='robot_bringup', executable='service_manager', output='screen'),
    ])
