"""Two IMX219s as independent Argus mono cameras (no hardware sync on this pair).

Publishes /left/image_raw, /right/image_raw (+ camera_info) via NITROS, plus
640x480 JPEG previews on /<side>/preview/image/compressed for remote viewing.
Sensor 1 is left, sensor 0 is right (checked on the robot); override with left_id/right_id.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode

# NITROS nodes agree on image formats (rgb8 / nv12 ...) during this window after they
# start. The default 1 s is shorter than an Argus camera takes to open, so a camera
# and its resizer could settle on different formats and no preview frames would flow.
NEGOTIATION_S = 12


def argus_mono(side, camera_id, mode, calib):
    return ComposableNode(
        name=f'argus_{side}',
        package='isaac_ros_argus_camera',
        plugin='nvidia::isaac_ros::argus::ArgusMonoNode',
        namespace='',
        parameters=[{
            'camera_id': camera_id,
            'type_negotiation_duration_s': NEGOTIATION_S,
            'mode': mode,
            'camera_link_frame_name': 'stereo_camera',
            'optical_frame_name': f'stereo_camera_{side}_optical',
            'camera_info_url': 'file://' + os.path.join(calib, f'{side}.yaml'),
        }],
        # The mono node always publishes under left/; route it to this side.
        remappings=[
            ('left/image_raw', f'{side}/image_raw'),
            ('left/camera_info', f'{side}/camera_info'),
            # Its own camera->optical TF has a garbage (unnormalised) rotation and
            # can't be turned off; the URDF publishes these frames instead.
            ('/tf_static', f'/argus_{side}/tf_static_unused'),
        ],
    )


def preview_resize(side):
    # GPU downscale in the camera container (NITROS zero-copy) for remote viewing.
    return ComposableNode(
        name=f'preview_resize_{side}',
        package='isaac_ros_image_proc',
        plugin='nvidia::isaac_ros::image_proc::ResizeNode',
        namespace='',
        parameters=[{
            'type_negotiation_duration_s': NEGOTIATION_S,
            'input_width': 1640,
            'input_height': 1232,
            'output_width': 640,
            'output_height': 480,
            # 1640x1232 is already 4:3. keep_aspect_ratio gives an odd-sized output, which
            # nv12 (if NITROS negotiates it) can't hold: the resizer fails and the container dies.
            'keep_aspect_ratio': False,
        }],
        remappings=[
            ('image', f'{side}/image_raw'),
            ('camera_info', f'{side}/camera_info'),
            ('resize/image', f'{side}/preview/image'),
            ('resize/camera_info', f'{side}/preview/camera_info'),
        ],
    )


def preview_jpeg(side):
    # No NVENC on Orin Nano, so the preview goes out as CPU JPEG.
    return Node(
        package='image_transport',
        executable='republish',
        name=f'preview_jpeg_{side}',
        arguments=['raw', 'compressed'],
        parameters=[{'out.jpeg_quality': 60}],  # ~1/3 the size of the default 95
        remappings=[
            ('in', f'{side}/preview/image'),
            ('out/compressed', f'{side}/preview/image/compressed'),
        ],
        condition=IfCondition(LaunchConfiguration('preview')),
    )


def side_nodes(context):
    calib = os.path.join(get_package_share_directory('robot_bringup'), 'config', 'calibration')
    choice = LaunchConfiguration('side').perform(context)
    sides = ('left', 'right') if choice == 'both' else (choice,)
    ids = {'left': LaunchConfiguration('left_id'), 'right': LaunchConfiguration('right_id')}
    nodes = []
    for side in sides:
        nodes += [argus_mono(side, ids[side], LaunchConfiguration('mode'), calib), preview_resize(side)]
    return [
        ComposableNodeContainer(
            name=f'camera_container_{choice}',
            namespace='',
            package='rclcpp_components',
            executable='component_container_mt',
            respawn=True,
            respawn_delay=3.0,
            composable_node_descriptions=nodes,
            output='screen',
        ),
        *[preview_jpeg(side) for side in sides],
    ]


def generate_launch_description():
    return LaunchDescription([
        # One camera or both. On the robot, camera_manager.py starts each side when it's needed.
        DeclareLaunchArgument('side', default_value='both'),
        DeclareLaunchArgument('left_id', default_value='1'),
        DeclareLaunchArgument('right_id', default_value='0'),
        # IMX219 modes: 0=3280x2464@21 1=3280x1848@28 2=1920x1080@30 3=1640x1232@30 4=1280x720@60
        DeclareLaunchArgument('mode', default_value='3'),
        DeclareLaunchArgument('preview', default_value='true'),
        OpaqueFunction(function=side_nodes),
    ])
