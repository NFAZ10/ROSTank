"""Full robot: Pico drive + lidar, stereo cameras, Foxglove bridge for remote viewing.

Connect Foxglove (app or https://app.foxglove.dev) to ws://<jetson-ip>:8765.
rosbridge (roslibjs pages, phone apps) on ws://<jetson-ip>:9090.
Mapping/navigation start and stop through the /robot/* services (nav_manager.py).
Web UI on http://<jetson-ip>:8080 runs separately (robot-web.service), so it can stop/start this.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Topics Foxglove may subscribe to. Raw camera images (~6 MB/frame) are left out
# on purpose: one click on them would saturate Wi-Fi.
FOXGLOVE_TOPICS = [
    '/scan', '/odom', '/wheel/odom', '/odom_rf2o', '/battery', '/battery/status', '/tf', '/tf_static', '/cmd_vel', '/rosout',
    '/(left|right)/preview/image/compressed', '/(left|right)/preview/camera_info',
    '/(left|right)/camera_info', '/diagnostics', '/robot_description',
    # Nav2 / SLAM, when robot_navigation is running
    '/map', '/plan', '/local_plan', '/(local|global)_costmap/costmap', '/local_costmap/published_footprint',
    '/amcl_pose', '/goal_pose', '/initialpose', '/robot/nav_mode',
    # Stereo depth: computed only while something subscribes (stereo_depth.py)
    '/depth/points', '/depth/image', '/depth/camera_info', '/depth/preview/compressed',
]


def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory('robot_bringup'), 'launch')
    description_dir = os.path.join(get_package_share_directory('robot_description'), 'launch')

    def include(name, directory=launch_dir):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(directory, name)))

    return LaunchDescription([
        DeclareLaunchArgument('foxglove', default_value='true'),
        DeclareLaunchArgument('rosbridge', default_value='true'),
        include('drive.launch.py'),
        include('cameras.launch.py'),
        # Sensor mounts come from the URDF (robot_description/urdf/robot.urdf.xacro).
        include('description.launch.py', description_dir),
        # Stereo depth from the camera previews, only while something watches it.
        Node(package='robot_bringup', executable='stereo_depth', output='screen'),
        # Mapping / navigation on demand (Foxglove Service Call buttons on /robot/*).
        Node(package='robot_bringup', executable='nav_manager', output='screen'),
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            parameters=[{
                'port': 8765,
                'topic_whitelist': FOXGLOVE_TOPICS,
                'send_buffer_limit': 20000000,
                'max_qos_depth': 5,
            }],
            condition=IfCondition(LaunchConfiguration('foxglove')),
        ),
        # rosbridge WebSocket + rosapi (topic/service listing that rosbridge clients use).
        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            parameters=[{'port': 9090}],
            condition=IfCondition(LaunchConfiguration('rosbridge')),
        ),
        Node(
            package='rosapi',
            executable='rosapi_node',
            condition=IfCondition(LaunchConfiguration('rosbridge')),
        ),
    ])
