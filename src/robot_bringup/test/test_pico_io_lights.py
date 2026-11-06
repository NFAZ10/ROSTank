import time

from robot_bringup.pico_io import (AMBER, BLINK, BODY, DEFAULT_COLORS, FRONT, HEAD, OFF, ORANGE_WHITE, SOLID, WIGWAG,
                                   WIGWAG2, PicoIO, rgbw)


def lights(**state):
    """A PicoIO with the shipped LED layout and no ROS node behind it."""
    n = PicoIO.__new__(PicoIO)
    n.counts, n.headlight_groups, n.warning_groups = [14, 24, 0, 8], [HEAD], []
    n.front_group = FRONT
    n.zones = {'sides': 'auto', 'back': 'auto', 'front': 'auto'}
    n.colors = dict(DEFAULT_COLORS)
    n.body_order = ['right', 'back', 'left']
    n.headlights, n.mode, n.battery = 'off', 'idle', ''
    n.docking = False
    n.v = n.w = 0.0
    n.cmd_time = time.monotonic()
    n.brake_until = 0.0
    for k, v in state.items():
        setattr(n, k, v)
    return n.segments()


RIGHT, BACK, LEFT, BAR = (BODY, 0, 8), (BODY, 8, 8), (BODY, 16, 8), (FRONT, 0, 8)


def test_body_strips_split_right_back_left():
    out = lights()
    assert set(out) == {(HEAD, 0, 14), RIGHT, BACK, LEFT, BAR}
    assert out[BAR][0] == OFF
    assert out[BACK] == (SOLID, (50, 0, 0, 0), 1000)


def test_turning_left_blinks_left_strip():
    out = lights(v=0.2, w=0.6)
    assert out[LEFT] == (BLINK, AMBER, 700)
    assert out[RIGHT][0] == SOLID


def test_reversing_shows_white_on_back():
    assert lights(v=-0.1)[BACK] == (SOLID, (0, 0, 0, 255), 1000)


def test_warning_while_navigating_flashes_orange_white_over_turn_signal():
    out = lights(mode='navigating', v=0.2, w=0.6)
    assert out[RIGHT] == out[LEFT] == out[BAR] == (WIGWAG2, ORANGE_WHITE, 800)
    assert out[BACK] == (WIGWAG, AMBER, 1000)


def test_zone_modes_override_auto():
    out = lights(mode='navigating', zones={'sides': 'off', 'back': 'on', 'front': 'on'})
    assert out[RIGHT][0] == out[LEFT][0] == OFF
    assert out[BACK] == (SOLID, (255, 0, 0, 0), 1000)
    assert out[BAR] == (SOLID, (0, 0, 0, 255), 1000)


def test_headlights_on_use_white_chip():
    assert lights(headlights='on')[(HEAD, 0, 14)] == (SOLID, (0, 0, 0, 255), 1000)


def test_rgbw_puts_white_on_the_white_chip():
    assert rgbw('#ffffff') == (0, 0, 0, 255)
    assert rgbw('#ff8040') == (191, 64, 0, 64)
    assert rgbw('#ffffff', 60 / 255) == (0, 0, 0, 60)


def test_on_uses_the_zone_colour():
    colors = dict(DEFAULT_COLORS, sides='#00ff00', headlights='#ff0000')
    out = lights(zones={'sides': 'on', 'back': 'on', 'front': 'auto'}, colors=colors, headlights='dim')
    assert out[LEFT] == out[RIGHT] == (SOLID, (0, 255, 0, 0), 1000)
    assert out[BACK] == (SOLID, (255, 0, 0, 0), 1000)  # default back colour: red
    assert out[(HEAD, 0, 14)] == (SOLID, (60, 0, 0, 0), 1000)


def run_charger(volts, count=2):
    from robot_bringup.pico_io import charger_state
    state, streak, out = None, 0, []
    for v in volts:
        state, streak = charger_state(state, streak, v, 9.0, 6.0, count)
        out.append(state)
    return out


def test_charger_needs_two_readings_to_change():
    assert run_charger([0.0, 14.5, 14.5, 14.5]) == [False, False, True, True]


def test_charger_ignores_a_single_blip_and_has_hysteresis():
    assert run_charger([0.0, 14.5, 0.0, 0.0]) == [False, False, False, False]
    assert run_charger([14.5, 7.0, 7.0, 5.0, 5.0]) == [True, True, True, True, False]


def test_docking_flashes_like_autonomous_driving():
    out = lights(docking=True)
    assert out[RIGHT] == out[LEFT] == out[BAR] == (WIGWAG2, ORANGE_WHITE, 800)
