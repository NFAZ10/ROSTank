from types import SimpleNamespace

import pytest
from pymavlink import mavutil

from robot_bringup.apm_imu import G, booted, imu_values, to_flu


def raw(**kw):
    values = dict(xgyro=0, ygyro=0, zgyro=0, xacc=0, yacc=0, zacc=0)
    values.update(kw)
    return SimpleNamespace(**values)


def test_axes_forward_right_down_to_forward_left_up():
    assert to_flu(1, 2, 3) == (1, -2, -3)


def test_flat_board_at_rest_reads_gravity_up():
    # ArduPilot reports specific force: at rest, z (down) reads -1 g.
    _, accel = imu_values(raw(zacc=-1000))
    assert accel == (0.0, -0.0, pytest.approx(G))


def test_units():
    gyro, accel = imu_values(raw(xgyro=500, yacc=-1000))
    assert gyro[0] == pytest.approx(0.5)
    assert accel[1] == pytest.approx(G)


def test_yaw_right_is_negative_in_ros():
    # Turning right (clockwise from above) is +z in down-axis terms, -z in ROS.
    gyro, _ = imu_values(raw(zgyro=100))
    assert gyro[2] == pytest.approx(-0.1)


@pytest.mark.parametrize('state,ready', [
    (mavutil.mavlink.MAV_STATE_BOOT, False),
    (mavutil.mavlink.MAV_STATE_CALIBRATING, False),
    (mavutil.mavlink.MAV_STATE_STANDBY, True),
    (mavutil.mavlink.MAV_STATE_CRITICAL, True),  # no RC transmitter: its normal state here
])
def test_booted(state, ready):
    assert booted(SimpleNamespace(system_status=state)) is ready
