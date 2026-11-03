import math

import numpy as np

from robot_bringup.depth_obstacles import ObstacleGrid, depth_to_base, from_odom, select_band, to_odom


def blob(x, y, n=40):
    return np.column_stack([np.full(n, x), np.full(n, y)])


def test_one_frame_is_not_enough():
    g = ObstacleGrid(confirm=2, window=3)
    g.update(blob(1.0, 0.0), 0.0)
    assert len(g.cells(0.0)) == 0


def test_seen_twice_is_confirmed_at_the_right_place():
    g = ObstacleGrid(cell=0.08, confirm=2, window=3)
    g.update(blob(1.01, 0.01), 0.0)
    g.update(blob(1.01, 0.01), 0.1)
    c = g.cells(0.1)
    assert len(c) == 1
    assert abs(c[0, 0] - 1.01) <= 0.04 and abs(c[0, 1] - 0.01) <= 0.04


def test_flicker_outside_the_window_never_confirms():
    g = ObstacleGrid(confirm=2, window=3)
    for i in range(12):
        g.update(blob(1.0, 0.0) if i % 4 == 0 else np.zeros((0, 2)), i * 0.1)
    assert len(g.cells(1.2)) == 0


def test_too_few_points_are_ignored():
    g = ObstacleGrid(min_points=15)
    for i in range(3):
        g.update(blob(1.0, 0.0, n=5), i * 0.1)
    assert len(g.cells(0.2)) == 0


def test_confirmed_cell_is_held_then_forgotten():
    g = ObstacleGrid(hold=2.0)
    g.update(blob(1.0, 0.0), 0.0)
    g.update(blob(1.0, 0.0), 0.1)
    g.update(np.zeros((0, 2)), 1.0)
    assert len(g.cells(1.5)) == 1
    assert len(g.cells(2.2)) == 0


def test_odom_round_trip():
    xy = np.array([[1.0, 0.5], [-0.3, 2.0]])
    pose = (2.0, -1.0, 0.7)
    assert np.allclose(from_odom(to_odom(xy, pose), pose), xy)


def test_to_odom_applies_heading():
    assert np.allclose(to_odom(np.array([[1.0, 0.0]]), (1.0, 2.0, math.pi / 2)), [[1.0, 3.0]])


def test_obstacle_stays_put_in_odom_while_driving_towards_it():
    g = ObstacleGrid(cell=0.08, confirm=2, window=3)
    for i, x in enumerate((0.0, 0.03)):  # 3 cm of travel between frames, wall at odom x = 1.01
        g.update(to_odom(blob(1.01 - x, 0.0), (x, 0.0, 0.0)), i * 0.1)
    assert len(g.cells(0.1)) == 1


def test_depth_projection_and_band():
    # Optical (z forward, y down) into base (x forward, z up), lens 0.07 m up, 0.13 m forward.
    to_base = np.array([[0, 0, 1, 0.13], [-1, 0, 0, 0], [0, -1, 0, 0.07], [0, 0, 0, 1.0]])
    depth = np.zeros((480, 640), np.float32)
    depth[140:142, 320:322] = 1.0  # 40 rows above the 180 centre row, 1 m out
    p = depth_to_base(depth, 500.0, 500.0, 320.0, 180.0, to_base, step=1)
    assert len(p) == 4
    assert np.allclose(p[0], [1.13, 0.0, 0.07 + 40 / 500.0])
    assert len(select_band(p, 0.05, 0.30, 1.8)) == 4
    assert len(select_band(p, 0.20, 0.30, 1.8)) == 0   # under the band
    assert len(select_band(p, 0.05, 0.30, 1.0)) == 0   # out of range
