import math
from unittest import mock

import pytest

from robot_bringup.battery_filter import BatteryFilter, charge
from robot_bringup.battery_standby import parse_battery


def test_charge_curve():
    assert charge(11.0) == 0.0
    assert charge(14.0) == 1.0
    assert charge(13.0) == pytest.approx(0.30)
    assert charge(13.25) == pytest.approx(0.80)


def test_reading_scales_and_averages():
    f = BatteryFilter()
    assert f.message(0.0, 2.0, 5.0, 1.0).voltage == pytest.approx(10.0)
    assert f.message(1.0, 3.0, 5.0, 1.0).voltage == pytest.approx(12.5)


def test_old_readings_drop_out_after_10_s():
    f = BatteryFilter()
    f.message(0.0, 2.0, 5.0, 1.0)
    assert f.message(11.0, 3.0, 5.0, 1.0).voltage == pytest.approx(15.0)


def test_floating_pin_means_not_connected():
    f = BatteryFilter()
    f.message(0.0, 2.5, 5.0, 1.0)
    m = f.message(0.5, 0.7, 5.0, 1.0)
    assert m.present is False and math.isnan(m.voltage)
    assert not f.hist


@pytest.mark.parametrize('line,pin', [('B 2392', 2.392), ('B 0', 0.0), ('E 1.0 2.0', None), ('B', None),
                                      ('B x', None), ('', None), ('garbage B 12', None)])
def test_parse_battery(line, pin):
    assert parse_battery(line) == pin


@pytest.mark.parametrize('state,down', [('inactive', True), ('failed', True), ('active', False),
                                        ('activating', False), ('deactivating', False)])
def test_port_only_taken_when_stack_is_really_down(state, down):
    from robot_bringup import battery_standby
    with mock.patch.object(battery_standby.subprocess, 'run', return_value=mock.Mock(stdout=state + '\n')):
        assert battery_standby.stack_is_down() is down
