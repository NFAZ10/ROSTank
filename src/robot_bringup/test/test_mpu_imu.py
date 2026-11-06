import math

import pytest

from robot_bringup.mpu_imu import GyroBias
from robot_bringup.pico_io import imu_from_counts


def fill(bias, gyro, n=100, moving=False):
    for _ in range(n):
        bias.add(gyro, moving)


def test_no_bias_until_a_full_quiet_window():
    b = GyroBias()
    fill(b, (0.1, 0.2, 0.3), n=99)
    assert b.bias is None
    b.add((0.1, 0.2, 0.3), False)
    assert b.bias == pytest.approx((0.1, 0.2, 0.3))


def test_correct_removes_bias():
    b = GyroBias()
    fill(b, (-0.185, 0.017, -0.021))
    assert b.correct((-0.185, 0.017, 0.479)) == pytest.approx((0.0, 0.0, 0.5))


def test_shaking_window_is_ignored():
    b = GyroBias()
    for i in range(100):
        b.add((0.0, 0.0, 0.5 if i % 2 else -0.5), False)
    assert b.bias is None


def test_motion_restarts_the_window():
    b = GyroBias()
    fill(b, (0.1, 0.1, 0.1), n=99)
    b.add((1.0, 1.0, 1.0), True)
    fill(b, (0.1, 0.1, 0.1), n=99)
    assert b.bias is None


def test_later_windows_follow_slowly():
    b = GyroBias(blend=0.1)
    fill(b, (0.0, 0.0, 0.1))
    fill(b, (0.0, 0.0, 0.2))
    assert b.bias[2] == pytest.approx(0.11)


def test_counts_to_si():
    gyro, accel = imu_from_counts([655, 0, -655, 4096, 0, -8192])
    assert gyro == pytest.approx((math.radians(10.0), 0.0, -math.radians(10.0)))
    assert accel == pytest.approx((9.80665, 0.0, -2 * 9.80665))
