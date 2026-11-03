import signal
import time
from unittest.mock import MagicMock

from robot_bringup.camera_manager import Proc


def proc(cmd=('sleep', '30')):
    return Proc('test', list(cmd), MagicMock())


def wait_for(cond, seconds=3.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end and not cond():
        time.sleep(0.02)
    return cond()


def test_starts_only_when_wanted():
    p = proc()
    p.update(False)
    assert p.proc is None
    p.update(True)
    assert p.running
    p.stop_now()


def test_stop_signals_then_clears():
    p = proc()
    p.update(True)
    child = p.proc
    p.update(False)
    assert not p.running
    assert wait_for(lambda: child.poll() is not None)
    assert child.returncode == -signal.SIGINT
    p.update(False)
    assert p.proc is None


def test_does_not_start_again_until_the_old_one_is_gone():
    p = proc(('sh', '-c', 'trap "" INT; sleep 30'))
    p.update(True)
    child = p.proc
    time.sleep(0.2)
    p.update(False)
    p.update(True)   # wanted again while still stopping
    assert p.proc is child and not p.running
    p.kill_at = 0.0  # hung: escalate
    p.update(True)
    assert wait_for(lambda: child.poll() is not None)
    assert child.returncode == -signal.SIGKILL


def test_crash_retries_after_a_pause():
    p = proc(('false',))
    p.update(True)
    first = p.proc
    assert wait_for(lambda: first.poll() is not None)
    p.update(True)
    assert p.proc is None  # not straight away
    p.retry_at = 0.0
    p.update(True)
    assert p.proc is not None and p.proc is not first
    p.stop_now()
