"""Nav2 for the tracked robot.

  ros2 launch robot_navigation navigation.launch.py                      # map while navigating (SLAM)
  ros2 launch robot_navigation navigation.launch.py map:=~/robot_ws/maps/room.yaml   # saved map + AMCL
      initial_pose:=x,y,yaw   (map frame) starts AMCL already localized there
      If room.keepout.yaml exists (no-go zones from the web map editor), both costmaps
      get Nav2's keepout filter: the planner routes around those areas and never ends in one.

Goals: "2D Goal Pose" in RViz (view.launch.py). With a saved map, first set
"2D Pose Estimate" where the robot actually is. Add use_sim_time:=true for the sim
(and export ROS_DOMAIN_ID=1).
"""
import os
import tempfile

import yaml

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction, TimerAction
from launch.conditions import UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def bringup(context):
    share = get_package_share_directory('robot_navigation')
    nav2_launch = os.path.join(get_package_share_directory('nav2_bringup'), 'launch')
    params = os.path.join(share, 'config', 'nav2.yaml')
    sim_time = LaunchConfiguration('use_sim_time').perform(context)
    map_yaml = os.path.expanduser(LaunchConfiguration('map').perform(context))
    initial_pose = LaunchConfiguration('initial_pose').perform(context)
    keepout_yaml = map_yaml[:-len('.yaml')] + '.keepout.yaml' if map_yaml.endswith('.yaml') else ''
    keepout = bool(keepout_yaml) and os.path.exists(keepout_yaml)
    if map_yaml and (initial_pose or keepout):
        with open(params) as f:
            config = yaml.safe_load(f)
        if initial_pose:
            x, y, yaw = (float(v) for v in initial_pose.split(','))
            amcl = config['amcl']['ros__parameters']
            amcl['set_initial_pose'] = True
            amcl['initial_pose'] = {'x': x, 'y': y, 'z': 0.0, 'yaw': yaw}
        if keepout:
            for name in ('global_costmap', 'local_costmap'):
                costmap = config[name][name]['ros__parameters']
                costmap['filters'] = ['keepout_filter']
                costmap['keepout_filter'] = {'plugin': 'nav2_costmap_2d::KeepoutFilter', 'enabled': True,
                                             'filter_info_topic': '/costmap_filter_info'}
        out = tempfile.NamedTemporaryFile('w', prefix='nav2_', suffix='.yaml', delete=False)
        yaml.safe_dump(config, out, default_flow_style=False)  # no aliases: rcl rejects them
        out.close()
        params = out.name

    if map_yaml:
        localization = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(nav2_launch, 'localization_launch.py')),
            launch_arguments={'map': map_yaml, 'params_file': params, 'use_sim_time': sim_time}.items())
    else:
        localization = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, 'launch', 'slam.launch.py')),
            launch_arguments={'use_sim_time': sim_time}.items())

    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2_launch, 'navigation_launch.py')),
        launch_arguments={'params_file': params, 'use_sim_time': sim_time}.items())
    actions = [localization, navigation]
    if keepout:
        actions += keepout_servers(keepout_yaml, sim_time == 'true')
    # Lidar odometry (scan-to-scan matching) for the EKF in robot_bringup/drive.launch.py:
    # corrects track slip, mostly in turns. Here rather than in drive.launch so the
    # lidar can stop spinning when nothing is navigating. The sim has perfect odometry.
    laser_odometry = Node(
        package='rf2o_laser_odometry',
        executable='rf2o_laser_odometry_node',
        name='rf2o_laser_odometry',
        parameters=[{
            'laser_scan_topic': '/scan',
            'odom_topic': '/odom_rf2o',
            'publish_tf': False,
            'base_frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'init_pose_from_topic': '',
            'freq': 10.0,
            'pose_covariance_xy': 0.01,
            'pose_covariance_yaw': 0.005,
        }],
        condition=UnlessCondition(LaunchConfiguration('use_sim_time')),
        # It logs every scan at INFO.
        ros_arguments=['--log-level', 'warn'],
        output='screen',
    )
    try:
        get_package_share_directory('rf2o_laser_odometry')
    except PackageNotFoundError:  # not built: EKF stays wheel-only
        return actions + [LogInfo(msg='rf2o_laser_odometry not installed: no lidar odometry')]
    return actions + [laser_odometry]


def keepout_servers(mask_yaml, sim_time):
    """Serve the no-go mask (its own map_server) and tell the costmaps' keepout filter about it."""
    return [
        Node(
            package='nav2_map_server', executable='map_server', name='filter_mask_server',
            parameters=[{'yaml_filename': mask_yaml, 'topic_name': '/keepout_filter_mask',
                         'frame_id': 'map', 'use_sim_time': sim_time}],
            output='screen'),
        Node(
            package='nav2_map_server', executable='costmap_filter_info_server', name='costmap_filter_info_server',
            parameters=[{'type': 0, 'filter_info_topic': '/costmap_filter_info',  # 0 = keepout
                         'mask_topic': '/keepout_filter_mask', 'base': 0.0, 'multiplier': 1.0,
                         'use_sim_time': sim_time}],
            output='screen'),
        # Started late: launched alongside all of Nav2 on the Jetson, its first get_state calls
        # time out before discovery settles and it gives up, leaving both servers inactive.
        TimerAction(period=8.0, actions=[Node(
            package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_costmap_filters',
            parameters=[{'autostart': True, 'node_names': ['filter_mask_server', 'costmap_filter_info_server'],
                         'use_sim_time': sim_time}],
            output='screen')]),
        LogInfo(msg=f'no-go zones from {mask_yaml}'),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value='', description='Saved map .yaml; empty = SLAM'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('initial_pose', default_value='', description='x,y,yaw in the map frame'),
        OpaqueFunction(function=bringup),
    ])
