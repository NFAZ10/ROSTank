"""RViz with the robot layout: model, TF, lidar, odometry and both camera previews.

Works against the real robot (domain 0) or a running sim (export ROS_DOMAIN_ID=1
and pass use_sim_time:=true first).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rviz_config = os.path.join(get_package_share_directory('robot_description'), 'rviz', 'robot.rviz')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('fixed_frame', default_value='odom'),
        Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', rviz_config, '-f', LaunchConfiguration('fixed_frame')],
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
            output='screen',
        ),
    ])
