# robot_description

URDF, RViz layout and Gazebo Fortress sim for the tracked robot.

| Command | What it does |
|---|---|
| `ros2 launch robot_description sim.launch.py` | Gazebo test room + RViz (needs the desktop, e.g. xrdp) |
| `ros2 launch robot_description sim.launch.py gui:=false rviz:=false foxglove:=true` | Headless sim; view in Foxglove at `ws://<jetson-ip>:8766` |
| `ROS_DOMAIN_ID=1 ros2 run teleop_twist_keyboard teleop_twist_keyboard` | Drive the sim (keep speeds ≤ 0.3 m/s, the real robot's top speed) |
| `ros2 launch robot_description view.launch.py` | RViz against the real robot |
| `ros2 launch robot_description model.launch.py` | Just the model in RViz, isolated (domain 2) |

- The sim runs on **ROS domain 1**; the real robot (robot.service) is on domain 0, so both can run
  at once. `export ROS_DOMAIN_ID=1` in any terminal that should talk to the sim.
- Same topics as the real robot: `/cmd_vel`, `/odom`, `/tf`, `/scan`, `/left|right/image_raw`,
  `/left|right/camera_info`, `/left|right/preview/image`.
- Tracks aren't simulated physically: the body moves straight from `/cmd_vel` and odometry is
  exact. Gazebo wheel physics made a poor track model. Real turning slip is calibrated on the
  robot (`track_width` in drive.yaml).
- **All dimensions are placeholders.** Edit the properties at the top of `urdf/robot.urdf.xacro`
  once the frame is built, and keep `wheel_radius` / `track_width` equal to
  `robot_bringup/config/drive.yaml`. The real robot reads its sensor mounts from this URDF.
