"""Pack voltage and charge from the divider at the Pico's GP26 (no node here: pico_bridge and battery_standby share it)."""
import collections

from sensor_msgs.msg import BatteryState

# LiFePO4 (4 cells) resting voltage -> charge. Under load it reads lower: a guide, not a gauge.
LFP_CURVE = ((12.0, 0.0), (12.8, 0.10), (12.9, 0.20), (13.0, 0.30), (13.1, 0.40),
             (13.2, 0.70), (13.3, 0.90), (13.6, 1.0))
NOT_CONNECTED_PIN_V = 1.5  # under ~8.3 V at the pack: divider not connected (a floating pin reads ~0.7 V)


def charge(volts):
    curve = LFP_CURVE
    if volts <= curve[0][0]:
        return 0.0
    if volts >= curve[-1][0]:
        return 1.0
    (v0, p0), (v1, p1) = next((a, b) for a, b in zip(curve, curve[1:]) if a[0] <= volts <= b[0])
    return p0 + (p1 - p0) * (volts - v0) / (v1 - v0)


class BatteryFilter:
    def __init__(self):
        self.hist = collections.deque()  # (monotonic time, volts)

    def message(self, now, pin_volts, divider, scale):
        """BatteryState for one reading, averaged over the last 10 s. The caller stamps it."""
        msg = BatteryState()
        msg.power_supply_technology = BatteryState.POWER_SUPPLY_TECHNOLOGY_LIFE
        msg.design_capacity = 12.0
        if pin_volts < NOT_CONNECTED_PIN_V:
            self.hist.clear()
            msg.present = False
            msg.voltage = msg.percentage = float('nan')
            return msg
        self.hist.append((now, pin_volts * divider * scale))
        while self.hist and now - self.hist[0][0] > 10.0:
            self.hist.popleft()
        volts = sum(v for _, v in self.hist) / len(self.hist)
        msg.present, msg.voltage, msg.percentage = True, float(volts), float(charge(volts))
        return msg
