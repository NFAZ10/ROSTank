"""One switchable service, started by service_manager.py: service:=imu | sensors | crash | foxglove | rosbridge | claude."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Topics Foxglove may subscribe to. Raw camera images (~6 MB/frame) are left out
# on purpose: one click on them would saturate Wi-Fi.
FOXGLOVE_TOPICS = [
    '/scan', '/odom', '/wheel/odom', '/odom_rf2o', '/battery', '/battery/status', '/dock/status',
    '/imu/data', '/imu/raw', '/imu/status', '/range/front', '/temperature', '/humidity', '/lights/status', '/tf', '/tf_static', '/cmd_vel', '/rosout',
    '/(left|right)/preview/image/compressed', '/(left|right)/preview/camera_info',
    '/(left|right)/camera_info', '/diagnostics', '/robot_description',
    # Nav2 / SLAM, when robot_navigation is running
    '/map', '/plan', '/local_plan', '/(local|global)_costmap/costmap', '/local_costmap/published_footprint',
    '/amcl_pose', '/goal_pose', '/initialpose', '/robot/nav_mode',
    # Stereo depth: computed only while something subscribes (stereo_depth.py)
    '/depth/points', '/depth/image', '/depth/camera_info', '/depth/preview/compressed',
]


def service_nodes(context):
    share = get_package_share_directory('robot_bringup')
    return {
        # Gyro bias removal for the IMU on the sensor Pico. Waits quietly if the sensors service is off.
        'imu': [Node(package='robot_bringup', executable='mpu_imu', output='screen')],
        # Sensor + LED Pico: front range, temperature/humidity, headlights / tail / top lights.
        'sensors': [Node(package='robot_bringup', executable='pico_io',
                         parameters=[os.path.join(share, 'config', 'pico_io.yaml')], output='screen')],
        # Crash detection: stops the robot and marks the spot on the map.
        'crash': [Node(package='robot_bringup', executable='crash_monitor', output='screen')],
        # Claude vision: after a crash (or when asked) sends a camera frame to the Claude API.
        'claude': [Node(package='robot_bringup', executable='claude_vision', output='screen')],
        'foxglove': [Node(package='foxglove_bridge', executable='foxglove_bridge', parameters=[{
            'port': 8765,
            'topic_whitelist': FOXGLOVE_TOPICS,
            'send_buffer_limit': 20000000,
            'max_qos_depth': 5,
        }])],
        # rosbridge WebSocket + rosapi (topic/service listing that rosbridge clients use).
        'rosbridge': [Node(package='rosbridge_server', executable='rosbridge_websocket', parameters=[{'port': 9090}]),
                      Node(package='rosapi', executable='rosapi_node')],
    }[LaunchConfiguration('service').perform(context)]


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('service'), OpaqueFunction(function=service_nodes)])
