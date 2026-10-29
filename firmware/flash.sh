#!/bin/bash
# Compile and flash the Pico drive firmware. No sudo or BOOTSEL press needed once
# the udev rules from jetson_cleanup.sh are installed (arduino-cli uses the 1200-baud
# reset to drop the Pico into BOOTSEL, then picotool loads it).
set -euo pipefail
cd "$(dirname "$0")/pico_drive"
CLI=~/.local/bin/arduino-cli
FQBN=rp2040:rp2040:rpipico
$CLI compile --fqbn $FQBN --output-dir build .
PT=$(find ~/.arduino15/packages/rp2040/tools -name picotool -type f | head -1)
if [ -e /dev/pico ]; then
  # 1200-baud touch drops the running firmware into BOOTSEL. Done here because
  # `arduino-cli upload` sometimes misses the reboot ("No drive to deploy").
  python3 -c "import serial, time; s = serial.Serial('/dev/pico', 1200); time.sleep(0.3); s.close()"
  for _ in $(seq 20); do lsusb -d 2e8a:0003 >/dev/null && break; sleep 0.5; done
fi
# Already in BOOTSEL also covers a fresh board, or the button held while plugging in.
"$PT" load -x build/pico_drive.ino.uf2
