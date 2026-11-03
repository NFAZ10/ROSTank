# Home-base AprilTag

Use tag36h11 ID 0. Measure the outer black square, excluding the white
margin; the default is 0.10 m. Mount it flat and stationary at home base,
visible to the left camera.

## Check detection without moving

With the robot stack running, run in a terminal on the Jetson:

```bash
cd ~/robot_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=0
python3 scripts/check_home_tag.py --tag-size 0.10
```

The check waits up to 20 seconds, reports the detected tag and estimated
camera-relative range, and exits. It sends no movement commands and needs
no rebuild. Use `--tag-size 0.15` if the measured black square is 15 cm.
It can also inspect a saved camera image with `--image /path/to/photo.jpg`.
Range estimates depend on the existing left-camera calibration.

## Map integration status

Detection alone does not establish that the robot is at home. The tag must
be associated with a pose in the particular saved map, then proximity and
orientation must be checked against that pose. A distant visible tag is
not proof that the robot has reached home.

The existing `robot_bringup/dock.py` draft is not registered in `setup.py`
or launched. It attempts autonomous approach, and requires review before
activation. The stationary check does not activate it or save a home pose.
