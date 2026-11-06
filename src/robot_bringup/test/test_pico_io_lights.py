import time

from robot_bringup.pico_io import AMBER, BLINK, BODY, CHASE, HEAD, SOLID, WIGWAG, PicoIO


def lights(**state):
    """A PicoIO with the shipped LED layout and no ROS node behind it."""
    n = PicoIO.__new__(PicoIO)
    n.counts, n.headlight_groups, n.warning_groups = [14, 24, 0, 0], [HEAD], []
    n.body_order = ['right', 'back', 'left']
    n.headlights, n.mode, n.battery = 'off', 'idle', ''
    n.v = n.w = 0.0
    n.cmd_time = time.monotonic()
    n.brake_until = 0.0
    for k, v in state.items():
        setattr(n, k, v)
    return n.segments()


RIGHT, BACK, LEFT = (BODY, 0, 8), (BODY, 8, 8), (BODY, 16, 8)


def test_body_strips_split_right_back_left():
    out = lights()
    assert set(out) == {(HEAD, 0, 14), RIGHT, BACK, LEFT}
    assert out[BACK] == (SOLID, (50, 0, 0, 0), 1000)


def test_turning_left_blinks_left_strip():
    out = lights(v=0.2, w=0.6)
    assert out[LEFT] == (BLINK, AMBER, 700)
    assert out[RIGHT][0] == SOLID


def test_reversing_shows_white_on_back():
    assert lights(v=-0.1)[BACK] == (SOLID, (0, 0, 0, 255), 1000)


def test_warning_while_navigating():
    out = lights(mode='navigating')
    assert out[RIGHT] == out[LEFT] == (CHASE, AMBER, 1200)
    assert out[BACK] == (WIGWAG, AMBER, 1000)


def test_headlights_on_use_white_chip():
    assert lights(headlights='on')[(HEAD, 0, 14)] == (SOLID, (0, 0, 0, 255), 1000)
