"""Spotting a crash from the IMU and tracks, and where to mark it (no ROS in here, so it can be tested).

Neither sign is enough alone: track vibration can jolt as hard as a soft bump (measured 3.9 m/s^2 driving,
2.7-7.1 for bumps at 0.15 m/s, 2026-11-03). A hit does both at once: a jolt against the direction of
travel, and the tracks suddenly falling behind what they're commanded (0.05-0.08 m/s for bumps, 0.03 or
less otherwise). Start-up and stopping don't count: the command must have been steady first.
"""
import json
import math
import os
from collections import deque


class CrashDetector:
    def __init__(self, jolt=2.5, lag_min=0.04, lag_fraction=0.25, window=0.3, steady=0.3, cooldown=1.0, samples=50):
        self.jolt_limit, self.lag_min, self.lag_fraction = jolt, lag_min, lag_fraction
        self.window, self.steady, self.cooldown = window, steady, cooldown
        self.recent = deque(maxlen=samples)
        self.cmd, self.cmd_since, self.wheel = 0.0, -math.inf, 0.0
        self.pending_until, self.pending_peak, self.last_crash = -math.inf, 0.0, -math.inf

    def on_cmd(self, t, v):
        if abs(v - self.cmd) > 0.03:
            self.cmd_since = t
            self.pending_until = -math.inf  # the driver changed something: not a hit we can judge
        self.cmd = v

    def on_wheel(self, v):
        self.wheel = v

    def on_accel(self, t, forward):
        """Forward acceleration in the robot's frame. Returns (direction, jolt) when a crash is confirmed."""
        baseline = sum(self.recent) / len(self.recent) if len(self.recent) == self.recent.maxlen else None
        self.recent.append(forward)
        if baseline is None or abs(self.cmd) < 0.05 or t - self.cmd_since < self.steady:
            return None
        direction = 1 if self.cmd > 0 else -1
        against = -direction * (forward - baseline)
        if against >= self.jolt_limit and t >= self.pending_until:
            self.pending_until, self.pending_peak = t + self.window, against
        elif t < self.pending_until:
            self.pending_peak = max(self.pending_peak, against)
        if t >= self.pending_until or t - self.last_crash < self.cooldown:
            return None
        behind = abs(self.cmd) - direction * self.wheel
        if behind >= max(self.lag_min, self.lag_fraction * abs(self.cmd)):
            self.last_crash, self.pending_until = t, -math.inf
            return direction, self.pending_peak
        return None


def mark_at(x, y, yaw, direction, half_length=0.16, reach=0.05):
    """The hit: just beyond the robot's front or back edge."""
    d = direction * (half_length + reach)
    return x + d * math.cos(yaw), y + d * math.sin(yaw)


def mark_points(marks, spread=0.08, z=0.15):
    """A small blob per mark, as costmap points: how big the object is can't be told from a bump."""
    return [(m['x'] + dx, m['y'] + dy, z) for m in marks
            for dx, dy in ((0, 0), (spread, 0), (-spread, 0), (0, spread), (0, -spread))]


def load_marks(path):
    try:
        with open(path) as f:
            marks = json.load(f)
        return marks if isinstance(marks, list) else []
    except (OSError, ValueError):
        return []


def save_marks(path, marks):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(marks, f)
    os.replace(tmp, path)


def remove_marks(marks, x, y, within=0.02):
    """The mark at (x, y); NaN means all of them."""
    if math.isnan(x):
        return []
    return [m for m in marks if math.hypot(m['x'] - x, m['y'] - y) > within]
