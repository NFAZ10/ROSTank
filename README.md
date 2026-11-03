# Tracked robot: command-line cheat sheet

SSH in (`ssh nfazio@<robot-ip>`) or open a terminal on the Jetson. Everything starts by
itself at boot. `robot help` lists every command.

## Everyday
```bash
robot status                 # is everything running? sensor rates, nav mode, addresses
robot start | stop | restart # the robot stack (drive, lidar, cameras, Foxglove, rosbridge)
robot logs nav               # follow logs: drive | nav | cameras | web (Ctrl-C quits)
robot power                  # Jetson power mode and current draw
```

## Driving (lift the robot or clear the floor)
```bash
robot drive                  # keyboard: i fwd, , back, j/l turn, k stop, q/z faster/slower
robot move 0.5               # 0.5 m forward (negative = back), optional speed: robot move 1 0.25
robot turn 90                # 90 deg left on the spot (negative = right)
robot stop-motors            # stop now + cancel navigation goals
robot wheels                 # live encoder counts/speeds per track
```
`move`/`turn` refuse while mapping or navigating, because Nav2 would fight them.

## Maps and navigation
```bash
robot map start              # SLAM: drive around (robot drive) to build the map
robot map explore            # or let it map on its own; comes back when done, then save
robot map continue kitchen   # keep adding to a saved map (start near where the robot last was on it), save as kitchen
robot map save kitchen       # name: letters, digits, - _
robot map list
robot map nav kitchen        # navigate on it (no name = latest). Starts where the robot last was.
robot pose 1.2 0.5 90        # if the robot was moved: where it really is (x m, y m, facing deg)
robot localize               # then turn on the spot until the lidar matches the map
robot findme                 # or: work out where it is from scratch (no pose needed)
robot dock                   # find the dock's AprilTag (tag36h11 id 0, 9 cm black square,
                             # upright, centre ~7 cm high) and stop 45 cm in front of it
robot goal 2.0 1.0 0         # drive there; prints metres to go; Ctrl-C cancels
robot where                  # current position
robot map stop
robot map delete kitchen
# hand-edit a saved map: web UI → Maps → Edit (or /mapedit); original kept, Revert restores it
```
Map coordinates: x/y in metres from where mapping started, angles in degrees, 0 = the
direction the robot faced at the start, positive = counter-clockwise. The web UI map shows
the same 1 m grid.

## Calibration and cameras
```bash
robot calib                  # wheel_radius, track_width, invert/swap flags
robot calib set track_width=0.19 wheel_radius=0.031   # live + saved to drive.yaml
robot snap left              # picture to ~/robot_snaps/ (left | right | depth)
robot depth                  # stereo distance to what's left / ahead / right
robot guard off | on         # obstacle protection for manual driving (or the Drive tab button)
```
Each camera runs only while something uses it (a view of it, depth, dock), so it takes about
20 s to start; `robot status` shows which are on. Stereo depth runs only while something watches it: web UI → Drive → **Depth** (red near,
blue far, black unknown), `robot depth`, or Foxglove on `/depth/points` or `/depth/image`.
It measures about 0.25–6 m, at about 4–8 fps.
Overriding protection affects only manual driving (joystick, `robot drive`), limits it to
0.15 m/s, and ends after 20 s without a manual command; Nav2 stays protected.
While driving, `drive_guard` also stops for `/depth/obstacles`: depth cut to 5–30 cm above the
floor and within 1.8 m, counted only once seen in 2 of 3 frames. After moving the cameras,
update `camera_pitch` (URDF) and `right_row_offset` (`stereo_depth`), or recalibrate.
The APM flight controller is a sensor board: `apm_imu` publishes its gyro and accelerometer on
`/imu/apm` at 50 Hz. Keep its servo outputs disconnected, and the robot still when it first powers up.
Drive calibration (1 m straight, 360° spin): web UI → Calibration.
Stereo camera calibration: web UI → Camera calibration (`/calibrate`).

## Battery
Wiring: 6.8 kΩ from battery + (after the fuse and switch) to Pico **GP26** (pin 31), and
1.5 kΩ from GP26 to GND. The web UI header shows voltage and approximate charge.
```bash
robot battery                        # live voltage and charge
robot calib set battery_scale=1.012  # trim: multimeter volts / shown volts
```
- **Low** (amber) below 12.4 V, and **critical** (red) below 12.0 V.
- Below **11.0 V for a minute**, the Jetson powers off cleanly, before the packs cut out.
- The charge percentage is from a LiFePO4 voltage curve. It reads low while driving, so trust it most at rest.

## Firmware and code
```bash
robot flash                  # rebuild + flash the Pico (stops/starts the stack around it)
robot build                  # colcon build the workspace (after editing src/)
```
Python, launch and config edits in `src/` take effect after `robot restart`; the workspace
is built with `--symlink-install`. A newly added file needs `robot build`.

## Talking to ROS directly
The `robot` command wraps these; use them to learn or debug:
```bash
ros2 node list                                   # what's running
ros2 topic list                                  # all data streams
ros2 topic hz /scan                              # how fast one arrives
ros2 topic echo /odom --once                     # one message
ros2 topic pub --once /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.0}}"   # stop
ros2 service call /robot/start_mapping std_srvs/srv/Trigger
ros2 param get /pico_bridge track_width
ros2 run tf2_ros tf2_echo map base_link          # robot pose
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
Key topics: `/cmd_vel` (drive command), `/odom` (fused odometry), `/wheel/odom` and
`/wheel/joint_states` (raw tracks), `/scan` (lidar), `/map`, `/left|right/image_raw` and
`/left|right/preview/image/compressed` (cameras), `/robot/nav_mode`.
Services: `/robot/start_mapping`, `/robot/save_map`, `/robot/start_navigation`, `/robot/stop`.

## Ways in
| What | Address |
|---|---|
| Web UI (map, drive, calibration, encoders) | `http://<robot-ip>:8080` |
| Foxglove Studio | `ws://<robot-ip>:8765` |
| rosbridge (roslibjs, phone apps) | `ws://<robot-ip>:9090` |

## Where things live
| | |
|---|---|
| Workspace / source | `~/robot_ws`, `~/robot_ws/src/{robot_bringup,robot_description,robot_navigation}` |
| Saved maps | `~/robot_ws/maps` |
| Drive settings | `src/robot_bringup/config/drive.yaml` |
| Camera calibration | `src/robot_bringup/config/calibration/{left,right}.yaml` |
| Pico firmware | `~/robot_ws/firmware/pico_drive/pico_drive.ino` |
| Services | `robot.service` (stack, `start_robot.sh`), `robot-web.service` (web UI, `start_web.sh`) |
| This tool | `~/robot_ws/scripts/robot` + `robot_cli.py` (linked as `~/.local/bin/robot`) |

The stack stops itself after 30 min with no driving, no mapping/navigation and no viewers
(web UI, Foxglove or rosbridge). `robot start` brings it back.
