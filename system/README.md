# System setup files

Copies of what lives outside the workspace, to rebuild the robot on a fresh Jetson:

```bash
sudo cp systemd/*.service /etc/systemd/system/ && sudo systemctl daemon-reload
sudo systemctl enable --now robot.service robot-web.service
sudo cp udev/*.rules /etc/udev/rules.d/ && sudo udevadm control --reload && sudo udevadm trigger
sudo install -m 440 sudoers/90-nfazio-nopasswd /etc/sudoers.d/   # passwordless sudo (web UI stack control)
```

- **`99-pico.rules`** gives the Pico drive board the name `/dev/pico`, and **`99-rplidar.rules`** gives the RPLIDAR A1 (CP2102 adapter) `/dev/rplidar`.
- **`99-usb-no-autosuspend.rules`** keeps USB devices from being power-saved.
- `src/rf2o_laser_odometry` is MAPIRlab's rf2o (GPL-3) with local fixes; see `docs/patches/`.
