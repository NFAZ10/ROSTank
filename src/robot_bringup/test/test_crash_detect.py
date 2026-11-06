import math

from robot_bringup.crash_detect import CrashDetector, load_marks, mark_at, mark_points, remove_marks, save_marks


def drive(det, t0=0.0, secs=2.0, v=0.15, a=0.0):
    t = t0
    while t < t0 + secs:
        det.on_cmd(t, v)
        det.on_wheel(v)
        assert det.on_accel(t, a) is None
        t += 0.02
    return t


def hit(det, t, jolt, v=0.15, wheel=0.09):
    det.on_cmd(t, v)
    det.on_wheel(wheel)
    return det.on_accel(t, jolt)


def test_jolt_with_tracks_falling_behind_is_a_crash():
    det = CrashDetector()
    t = drive(det)
    assert hit(det, t, -5.0) == (1, 5.0)


def test_reverse_crash_jolts_the_other_way():
    det = CrashDetector()
    t = drive(det, v=-0.15)
    assert hit(det, t, 3.0, v=-0.15, wheel=-0.09) == (-1, 3.0)


def test_jolt_without_track_lag_is_vibration():
    det = CrashDetector()
    t = drive(det)
    assert hit(det, t, -5.0, wheel=0.14) is None


def test_track_lag_without_a_jolt_is_nothing():
    det = CrashDetector()
    t = drive(det)
    assert hit(det, t, -1.0) is None


def test_jolt_along_travel_is_not_a_crash():
    det = CrashDetector()
    t = drive(det)
    assert hit(det, t, 5.0) is None


def test_starting_off_does_not_count():
    det = CrashDetector()
    t = drive(det, v=0.0)
    det.on_cmd(t, 0.15)
    det.on_wheel(0.0)
    assert det.on_accel(t + 0.1, -5.0) is None


def test_standing_still_does_not_count():
    det = CrashDetector()
    t = drive(det, v=0.0)
    assert hit(det, t, -5.0, v=0.0, wheel=0.0) is None


def test_lag_scales_with_speed():
    det = CrashDetector()
    t = drive(det, v=0.35)
    assert hit(det, t, -5.0, v=0.35, wheel=0.30) is None  # 0.05 behind: ordinary at this speed
    assert hit(det, t + 0.02, -5.0, v=0.35, wheel=0.2) is not None


def test_lag_may_arrive_just_after_the_jolt():
    det = CrashDetector()
    t = drive(det)
    assert hit(det, t, -5.0, wheel=0.15) is None
    assert hit(det, t + 0.1, -0.5, wheel=0.09) is not None
    t2 = drive(CrashDetector())
    late = CrashDetector()
    t2 = drive(late)
    assert hit(late, t2, -5.0, wheel=0.15) is None
    assert hit(late, t2 + 0.5, -0.5, wheel=0.09) is None  # too long after to be the same event


def test_one_crash_is_not_counted_twice():
    det = CrashDetector()
    t = drive(det)
    assert hit(det, t, -5.0) is not None
    assert hit(det, t + 0.5, -5.0) is None


def test_mark_is_beyond_the_edge_being_driven_into():
    assert mark_at(1.0, 2.0, 0.0, 1) == (1.0 + 0.21, 2.0)
    x, y = mark_at(1.0, 2.0, math.pi / 2, -1)
    assert abs(x - 1.0) < 1e-9 and abs(y - (2.0 - 0.21)) < 1e-9


def test_mark_blob_is_centred():
    pts = mark_points([{'x': 1.0, 'y': 2.0}])
    assert len(pts) == 5 and pts[0] == (1.0, 2.0, 0.15)


def test_marks_round_trip_and_survive_a_bad_file(tmp_path):
    path = str(tmp_path / 'm.crash.json')
    assert load_marks(path) == []
    save_marks(path, [{'x': 1.0, 'y': 2.0}])
    assert load_marks(path) == [{'x': 1.0, 'y': 2.0}]
    (tmp_path / 'bad.json').write_text('{oops')
    assert load_marks(str(tmp_path / 'bad.json')) == []


def test_remove_one_or_all():
    marks = [{'x': 1.0, 'y': 2.0}, {'x': 3.0, 'y': 4.0}]
    assert remove_marks(marks, 1.0, 2.0) == [{'x': 3.0, 'y': 4.0}]
    assert remove_marks(marks, 9.0, 9.0) == marks
    assert remove_marks(marks, math.nan, math.nan) == []
