import time

from robot_bringup.dock import Dock


class Ready:
    def __init__(self):
        self.calls = 0

    def service_is_ready(self):
        return True

    def call_async(self, req):
        self.calls += 1


def dock(tmp_path, **state):
    d = Dock.__new__(Dock)
    d.state, d.charger, d.auto_block_until, d.auto_pending = 'idle', False, 0.0, ''
    d.auto_battery, d.auto_idle_s, d.auto_retry_s = True, 300.0, 600.0
    d.battery, d.mode, d.nav_map, d.map_dir = 'ok', 'navigating', 'Office', str(tmp_path)
    d.last_motion = d.mode_since = time.monotonic()
    d.stop_exploring = Ready()
    d.started, d.said = [], []
    d.start_go_dock = lambda wait=2.0: (d.started.append(wait), (True, 'going to the dock'))[1]
    d.say = d.said.append
    d.get_logger = lambda: type('L', (), {'warn': lambda self, m: None})()
    for k, v in state.items():
        setattr(d, k, v)
    return d


def test_low_battery_goes_home(tmp_path):
    d = dock(tmp_path, battery='low: charge soon')
    d.auto_dock()
    assert d.started and 'battery low' in d.said[0]
    d.state = 'idle'
    d.auto_dock()  # no second try until auto_retry_s
    assert len(d.started) == 1


def test_not_while_charging(tmp_path):
    d = dock(tmp_path, battery='critical: charge now', charger=True)
    d.auto_dock()
    assert not d.started


def test_idle_needs_a_known_dock(tmp_path):
    d = dock(tmp_path)
    d.last_motion = d.mode_since = time.monotonic() - 400
    d.auto_dock()
    assert not d.started  # no Office.dock.json
    (tmp_path / 'Office.dock.json').write_text('{"x": 0, "y": 0, "yaw": 0}')
    d.auto_dock()
    assert d.started and 'idle' in d.said[0]


def test_recent_motion_is_not_idle(tmp_path):
    (tmp_path / 'Office.dock.json').write_text('{}')
    d = dock(tmp_path)
    d.mode_since = time.monotonic() - 400
    d.auto_dock()
    assert not d.started


def test_exploring_stops_the_explorer_first(tmp_path):
    d = dock(tmp_path, battery='low: charge soon', mode='exploring')
    d.auto_dock()
    assert d.stop_exploring.calls == 1 and not d.started
    d.mode = 'mapping'
    d.auto_dock()
    assert d.started
