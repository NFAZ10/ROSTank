"""Turns a stereo depth image into obstacles worth stopping for (no ROS in here, so it can be tested).

A single stereo frame can't be trusted on its own: mismatches on plain or shiny floor come
and go between frames, while anything real stays put. So points are cut to the height band
the robot could hit, binned into ground cells in the odom frame (so driving doesn't smear
them), and a cell only counts once it has been seen in `confirm` of the last `window` frames.
A confirmed cell is then held for `hold` seconds: low things drop out of the camera's view
as the robot gets close, and the lidar passes over them.
"""
from collections import Counter, deque

import numpy as np


def depth_to_base(depth, fx, fy, cx, cy, to_base, step=2):
    """Valid depth pixels (every step-th) as Nx3 points through the 4x4 to_base transform."""
    z = depth[::step, ::step]
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    ok = z > 0
    z, u, v = z[ok], u[ok], v[ok]
    pts = np.stack([(u - cx) * z / fx, (v - cy) * z / fy, z, np.ones_like(z)])
    return (to_base @ pts)[:3].T


def select_band(points, min_height, max_height, max_range):
    """Ground (x, y) of the points within the height band and range of the robot."""
    keep = (points[:, 2] > min_height) & (points[:, 2] < max_height) & \
        (np.hypot(points[:, 0], points[:, 1]) < max_range)
    return points[keep, :2]


def to_odom(xy, pose):
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack([x + c * xy[:, 0] - s * xy[:, 1], y + s * xy[:, 0] + c * xy[:, 1]], axis=1)


def from_odom(xy, pose):
    x, y, yaw = pose
    c, s = np.cos(yaw), np.sin(yaw)
    dx, dy = xy[:, 0] - x, xy[:, 1] - y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


class ObstacleGrid:
    def __init__(self, cell=0.08, min_points=15, confirm=2, window=3, hold=2.0):
        self.cell, self.min_points, self.confirm, self.hold = cell, min_points, confirm, hold
        self.recent = deque(maxlen=window)  # per frame: set of occupied cells
        self.confirmed = {}                 # cell -> last time it was seen

    def update(self, xy, now):
        """Add one frame's points (odom frame, metres; now in seconds)."""
        cells = set()
        if len(xy):
            ids, counts = np.unique(np.floor(xy / self.cell).astype(np.int64), axis=0, return_counts=True)
            cells = {tuple(c) for c in ids[counts >= self.min_points].tolist()}
        self.recent.append(cells)
        seen = Counter(c for frame in self.recent for c in frame)
        for c in cells:
            if seen[c] >= self.confirm:
                self.confirmed[c] = now

    def cells(self, now):
        """Centres of the confirmed cells (odom frame), Nx2. Drops those not seen for hold seconds."""
        self.confirmed = {c: t for c, t in self.confirmed.items() if now - t <= self.hold}
        if not self.confirmed:
            return np.zeros((0, 2))
        return (np.array(list(self.confirmed), float) + 0.5) * self.cell

    def clear(self):
        self.recent.clear()
        self.confirmed.clear()
