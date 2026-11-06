import math

from robot_bringup.claude_vision import prompt_for, read_key


def test_crash_prompt_mentions_the_hit_and_the_range():
    p = prompt_for('crash', 0.12)
    assert 'bumped' in p and '12 cm' in p


def test_clear_range_and_no_range():
    assert 'nothing within 2 m' in prompt_for('asked', math.inf)
    assert 'distance sensor' not in prompt_for('asked', None)


def test_key_file(tmp_path):
    assert read_key(str(tmp_path / 'missing')) is None
    (tmp_path / 'k').write_text('sk-ant-test\n')
    assert read_key(str(tmp_path / 'k')) == 'sk-ant-test'
