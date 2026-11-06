import math

import numpy as np
import pytest

from robot_bringup.dock import line_errors, steer


def tag(x, y, facing_deg):
    """Tag pose in base_link: at (x, y), its face pointing back at the robot, turned facing_deg."""
    a = math.radians(180 + facing_deg)
    T = np.eye(4)
    # Tag z (out of the face) in the floor plane; y down, x completes the frame.
    z = np.array([math.cos(a), math.sin(a), 0.0])
    yv = np.array([0.0, 0.0, -1.0])
    T[:3, 0], T[:3, 1], T[:3, 2] = np.cross(yv, z), yv, z
    T[:3, 3] = (x, y, 0.12)
    return T


def test_square_and_centred():
    d, lat, hdg = line_errors(tag(0.5, 0.0, 0))
    assert (d, lat, hdg) == pytest.approx((0.5, 0.0, 0.0), abs=1e-9)


def test_line_to_the_left():
    d, lat, hdg = line_errors(tag(0.5, 0.03, 0))
    assert lat == pytest.approx(0.03) and hdg == pytest.approx(0.0, abs=1e-9)
    assert steer(lat, hdg, 0.06) > 0  # turns left towards it


def test_robot_turned_right_of_the_dock():
    # Robot yawed 10 deg right: the tag's normal appears rotated left in base_link.
    d, lat, hdg = line_errors(tag(0.5, 0.0, 10))
    assert hdg == pytest.approx(math.radians(10))
    assert steer(0.0, hdg, 0.06) > 0


def test_lateral_offset_shifts_the_line():
    _, lat, _ = line_errors(tag(0.5, 0.01, 0), lateral_offset=0.01)
    assert lat == pytest.approx(0.0, abs=1e-9)


def test_steer_is_clamped():
    assert steer(0.5, 1.0, 0.06) == pytest.approx(0.4)


def test_intercept_is_capped_so_the_tag_stays_in_view():
    from robot_bringup.dock import steer
    # 10 cm off the line, already square: turns towards it, but only as hard as a 12 deg approach.
    assert steer(0.10, 0.0, 0.06, max_turn=1.0) == pytest.approx(2.0 * math.radians(12))
    # Already 12 deg towards the line: no more turning in.
    assert steer(0.10, -math.radians(12), 0.06) == pytest.approx(0.0, abs=1e-9)


def test_estimate_is_fixed_in_odom_and_averages_noise():
    from robot_bringup.dock import DockEstimate, line_errors_2d
    est = DockEstimate()
    rng = np.random.default_rng(1)
    for _ in range(30):  # robot still at the origin; tag 0.6 m ahead, normal noisy by +-3 deg
        a = math.radians(180 + rng.uniform(-3, 3))
        est.add((0.0, 0.0, 0.0), (0.6, 0.0), (math.cos(a), math.sin(a)))
    d, lat, hdg = line_errors_2d(*est.in_base((0.0, 0.0, 0.0)))
    assert abs(math.degrees(hdg)) < 1.5 and abs(lat) < 0.015
    # Robot drives 0.2 m forward and turns 10 deg left: the dock stays put.
    d, lat, hdg = line_errors_2d(*est.in_base((0.2, 0.0, math.radians(10))))
    assert d == pytest.approx(0.4 * math.cos(math.radians(10)), abs=0.02)
    assert math.degrees(hdg) == pytest.approx(-10, abs=1.5)
