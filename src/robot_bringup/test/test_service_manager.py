import json

from robot_bringup.service_manager import SERVICES, launch_command, load_choices


def test_everything_on_without_a_saved_file(tmp_path):
    assert load_choices(str(tmp_path / 'none.json')) == {name: name != 'claude' for name in SERVICES}


def test_saved_choice_is_kept(tmp_path):
    path = tmp_path / 's.json'
    path.write_text(json.dumps({'foxglove': False}))
    choices = load_choices(str(path))
    assert choices['foxglove'] is False
    assert choices['imu'] is True


def test_corrupt_file_means_all_on(tmp_path):
    path = tmp_path / 's.json'
    path.write_text('{not json')
    choices = load_choices(str(path))
    assert all(on for name, on in choices.items() if name != 'claude') and choices['claude'] is False


def test_only_a_clear_false_turns_a_service_off(tmp_path):
    path = tmp_path / 's.json'
    path.write_text(json.dumps({'imu': 0, 'sensors': None}))
    choices = load_choices(str(path))
    assert choices['imu'] and choices['sensors']


def test_safety_critical_nodes_cannot_be_switched():
    assert not {'drive', 'guard', 'drive_guard', 'lidar', 'odometry'} & set(SERVICES)


def test_launch_command_names_the_service():
    assert launch_command('imu')[-1] == 'service:=imu'


def test_claude_vision_is_off_until_switched_on(tmp_path):
    path = tmp_path / 's.json'
    assert load_choices(str(path))['claude'] is False
    path.write_text(json.dumps({'claude': True}))
    assert load_choices(str(path))['claude'] is True
