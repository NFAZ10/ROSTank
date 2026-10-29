"""Gazebo Fortress simulation of the tracked robot in a test room.

Same topics as the real robot (/cmd_vel, /odom, /tf, /scan, /left|right/...), but in
ROS domain 1 so it can run while robot.service drives the real one on domain 0.
To talk to it from another terminal:  export ROS_DOMAIN_ID=1

  ros2 launch robot_description sim.launch.py                 # Gazebo GUI + RViz (desktop)
  ros2 launch robot_description sim.launch.py gui:=false rviz:=false foxglove:=true   # over SSH
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, IncludeLaunchDescription,
                            OpaqueFunction, RegisterEventHandler, SetEnvironmentVariable)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def gazebo(context):
    share = get_package_share_directory('robot_description')
    world = LaunchConfiguration('world').perform(context)
    if not os.path.isabs(world):
        world = os.path.join(share, 'worlds', world)
    # -r: start unpaused. Without the GUI, render sensors off-screen via EGL.
    args = f'-r -v 2 {world}'
    if LaunchConfiguration('gui').perform(context) != 'true':
        args += ' -s --headless-rendering'
    # Run ign directly, not via ros_gz_sim's gz_sim.launch.py: that one starts it through a
    # shell, and Ctrl-C then kills only the shell and leaves Gazebo running in the background.
    gz = ExecuteProcess(cmd=['ign', 'gazebo'] + args.split(), output='screen', sigterm_timeout='5')
    return [gz, RegisterEventHandler(OnProcessExit(target_action=gz, on_exit=[EmitEvent(event=Shutdown())]))]


def generate_launch_description():
    share = get_package_share_directory('robot_description')
    launch_dir = os.path.join(share, 'launch')
    sim_time = {'use_sim_time': True}
    return LaunchDescription([
        DeclareLaunchArgument('domain', default_value='1'),
        DeclareLaunchArgument('world', default_value='test_room.sdf'),
        DeclareLaunchArgument('gui', default_value='true', description='Gazebo window (needs a desktop)'),
        DeclareLaunchArgument('rviz', default_value='true', description='RViz window (needs a desktop)'),
        DeclareLaunchArgument('foxglove', default_value='false', description='Foxglove bridge on port 8766'),
        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain')),
        OpaqueFunction(function=gazebo),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'description.launch.py')),
            launch_arguments={'sim': 'true', 'use_sim_time': 'true'}.items()),
        Node(
            package='ros_gz_sim',
            executable='create',
            arguments=['-topic', 'robot_description', '-name', 'tracked_robot', '-z', '0.005'],
            output='screen',
        ),
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            parameters=[{'config_file': os.path.join(share, 'config', 'gz_bridge.yaml')}, sim_time],
            output='screen',
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(launch_dir, 'view.launch.py')),
            launch_arguments={'use_sim_time': 'true'}.items(),
            condition=IfCondition(LaunchConfiguration('rviz'))),
        Node(
            package='foxglove_bridge',
            executable='foxglove_bridge',
            parameters=[{'port': 8766}, sim_time],
            condition=IfCondition(LaunchConfiguration('foxglove')),
        ),
    ])
