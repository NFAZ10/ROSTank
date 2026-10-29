# robot_navigation

slam_toolbox mapping + Nav2 for the tracked robot. Tested in the Gazebo sim.

## Real robot (robot.service already provides /scan, /odom, TF)

| Step | Command |
|---|---|
| Map + navigate | `ros2 launch robot_navigation navigation.launch.py` |
| Watch / send goals | RViz: `ros2 launch robot_description view.launch.py fixed_frame:=map`, then "2D Goal Pose". Or Foxglove (port 8765) |
| Save the map | `ros2 run nav2_map_server map_saver_cli -f ~/robot_ws/maps/<name>` |
| Navigate on a saved map | `ros2 launch robot_navigation navigation.launch.py map:=~/robot_ws/maps/<name>.yaml`, then "2D Pose Estimate" where the robot is |

The lidar motor starts by itself when SLAM/Nav2 subscribe to /scan (lidar_standby).

## Sim

```
ros2 launch robot_description sim.launch.py               # terminal 1
export ROS_DOMAIN_ID=1                                    # terminal 2
ros2 launch robot_navigation navigation.launch.py use_sim_time:=true
```

## Tuning notes

- `config/nav2.yaml`: Regulated Pure Pursuit (rotates in place toward the path), 0.22 m/s cruise,
  0.32 x 0.25 m footprint (chassis + 1 cm), 0.35 m inflation. Speeds are capped under the real top speed (0.3 m/s).
- `config/slam.yaml`: slam_toolbox online async, 5 cm map.
- Real-robot odometry will be the weak point: with no encoders yet it is integrated from
  commanded speeds, and tracks slip when turning. Calibrate `track_width` in drive.yaml with a
  360-degree spin test; an IMU (gyro) fused with robot_localization fixes heading properly.
- Check the lidar's forward direction in RViz on the real robot; if the scan is rotated, set a
  yaw on `laser_joint` in robot_description/urdf/robot.urdf.xacro.
