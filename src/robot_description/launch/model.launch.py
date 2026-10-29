"""Inspect the URDF on its own: robot_state_publisher + RViz, nothing else.

Runs in its own ROS domain (2) so it never mixes with the real robot or the sim.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory('robot_description'), 'launch')
    return LaunchDescription([
        DeclareLaunchArgument('domain', default_value='2'),
        DeclareLaunchArgument('sim', default_value='false', description='true loads the Gazebo sensor/plugin additions too'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain')),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'description.launch.py')),
            launch_arguments={'sim': LaunchConfiguration('sim')}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'view.launch.py')),
            launch_arguments={'fixed_frame': 'base_link'}.items()),
    ])
