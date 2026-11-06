#!/bin/bash
# Entry point for robot-battery.service: battery reading and low-battery shutdown, kept outside
# robot.service so they keep working while the stack is stopped.
source /opt/ros/humble/setup.bash
source "$HOME/robot_ws/install/setup.bash"
export ROS_DOMAIN_ID=0
exec ros2 launch robot_bringup battery.launch.py
