#!/bin/bash
# Entry point for robot.service: full robot bringup with ROS env.
source /opt/ros/humble/setup.bash
source "$HOME/robot_ws/install/setup.bash"
export ROS_DOMAIN_ID=0
exec ros2 launch robot_bringup robot.launch.py
