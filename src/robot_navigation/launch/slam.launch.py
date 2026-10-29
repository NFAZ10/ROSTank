"""Build a map with slam_toolbox from /scan + /odom. Publishes /map and map->odom.

Save it when done:  ros2 run nav2_map_server map_saver_cli -f ~/robot_ws/maps/<name>
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params = os.path.join(get_package_share_directory('robot_navigation'), 'config', 'slam.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            parameters=[params, {'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen',
        ),
    ])
