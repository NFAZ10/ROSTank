#!/bin/bash
# Entry point for robot-web.service: the web UI, kept outside robot.service so it
# can stop and start the robot stack.
source /opt/ros/humble/setup.bash
source "$HOME/robot_ws/install/setup.bash"
export ROS_DOMAIN_ID=0
exec ros2 run robot_bringup web_ui
