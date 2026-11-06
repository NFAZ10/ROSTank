#!/bin/bash
# Compile and flash Pico firmware. No sudo or BOOTSEL press needed once the udev rules
# from system/udev are installed (a 1200-baud touch drops the running firmware into
# BOOTSEL, then picotool loads it).
#
#   flash.sh [drive|io]   drive = pico_drive on /dev/pico (default), io = pico_io on /dev/pico_io
#
# Two Picos are attached, so it only loads when exactly one board is in BOOTSEL and
# that one is known to be the target. FQBN=... overrides the board (default: plain Pico).
set -euo pipefail
BOARD=${1:-drive}
case $BOARD in
  drive) SKETCH=pico_drive; PORT=/dev/pico;    OTHER=/dev/pico_io ;;
  io)    SKETCH=pico_io;    PORT=/dev/pico_io; OTHER=/dev/pico ;;
  *) echo "usage: $0 [drive|io]" >&2; exit 2 ;;
esac

# robot-battery.service reads /dev/pico while the stack is down, and anything holding the port blocks the reset.
if systemctl is-active -q robot-battery.service; then
  sudo -n systemctl stop robot-battery.service
  trap 'sudo -n systemctl start robot-battery.service' EXIT
fi
cd "$(dirname "$0")/$SKETCH"
CLI=~/.local/bin/arduino-cli
FQBN=${FQBN:-rp2040:rp2040:rpipico}
$CLI compile --fqbn "$FQBN" --output-dir build .
PT=$(find ~/.arduino15/packages/rp2040/tools -name picotool -type f | head -1)
bootsel() { lsusb -d 2e8a:0003 | wc -l; }
picos() { lsusb -d 2e8a: | wc -l; }

if [ "$(bootsel)" -gt 0 ] && [ -e "$PORT" ]; then
  echo "a Pico is already in BOOTSEL: can't tell it from $BOARD. Unplug it or flash it first." >&2; exit 1
fi
if [ -e "$PORT" ]; then
  # 1200-baud touch drops the running firmware into BOOTSEL. Done here because
  # `arduino-cli upload` sometimes misses the reboot ("No drive to deploy").
  python3 -c "import serial, time; s = serial.Serial('$PORT', 1200); time.sleep(0.3); s.close()"
  for _ in $(seq 40); do [ "$(bootsel)" -gt 0 ] && break; sleep 0.5; done
elif [ "$(bootsel)" -eq 1 ] && { [ -e "$OTHER" ] || [ "$(picos)" -eq 1 ]; }; then
  # A blank board, or the button held while plugging in: the other board is running
  # normally (or is not attached), so the one in BOOTSEL is this one.
  echo "$PORT not present: flashing the one Pico in BOOTSEL as $BOARD"
else
  echo "$PORT not found and can't tell which Pico is $BOARD ($(bootsel) in BOOTSEL, $(picos) attached)." >&2
  exit 1
fi
if [ "$(bootsel)" -ne 1 ]; then
  echo "expected exactly one Pico in BOOTSEL, found $(bootsel)" >&2; exit 1
fi
"$PT" load -x build/$SKETCH.ino.uf2
