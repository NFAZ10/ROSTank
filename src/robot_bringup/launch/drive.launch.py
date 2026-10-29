import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('robot_bringup')
    return LaunchDescription([
        Node(
            package='robot_bringup',
            executable='pico_bridge',
            parameters=[os.path.join(share, 'config', 'drive.yaml')],
            # Raw track odometry; the EKF below publishes the fused /odom + TF.
            # Drive commands arrive through drive_guard (obstacle stop), not straight from /cmd_vel.
            remappings=[('odom', 'wheel/odom'), ('cmd_vel', 'cmd_vel_safe')],
            output='screen',
        ),
        # Low-battery warning and clean power-off before the packs go flat (reads /battery).
        Node(package='robot_bringup', executable='battery_guard', output='screen'),
        # Every /cmd_vel (web UI, CLI, teleop, Nav2) -> obstacle check -> /cmd_vel_safe -> pico_bridge.
        Node(package='robot_bringup', executable='drive_guard', output='screen'),
        # Fuses /wheel/odom with lidar odometry (/odom_rf2o, only while mapping or
        # navigating) into /odom and odom->base_link. Without rf2o it is wheel-only.
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node',
            parameters=[os.path.join(share, 'config', 'ekf.yaml')],
            remappings=[('odometry/filtered', 'odom')],
            output='screen',
        ),
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            # Exits if the USB link drops; come back when it reappears.
            respawn=True,
            respawn_delay=3.0,
            parameters=[{
                'serial_port': '/dev/rplidar',
                'serial_baudrate': 115200,
                'frame_id': 'laser',
                'angle_compensate': True,
            }],
            output='screen',
        ),
        # Motor spins only while something consumes /scan (see lidar_standby.py).
        Node(
            package='robot_bringup',
            executable='lidar_standby',
            output='screen',
        ),
    ])
