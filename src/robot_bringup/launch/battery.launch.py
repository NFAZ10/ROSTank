"""Always on (robot-battery.service), whether or not the stack is running: battery reading and shutdown."""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='robot_bringup', executable='battery_standby', output='screen'),
        # Low-battery warning and clean power-off before the packs go flat (reads /battery).
        Node(package='robot_bringup', executable='battery_guard', output='screen'),
    ])
