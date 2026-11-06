from glob import glob

from setuptools import setup

package_name = 'robot_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/calibration', glob('config/calibration/*.yaml')),
        ('share/' + package_name + '/web', glob('web/*')),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'pico_bridge = robot_bringup.pico_bridge:main',
            'lidar_standby = robot_bringup.lidar_standby:main',
            'nav_manager = robot_bringup.nav_manager:main',
            'web_ui = robot_bringup.web_ui:main',
            'stereo_depth = robot_bringup.stereo_depth:main',
            'drive_guard = robot_bringup.drive_guard:main',
            'explorer = robot_bringup.explorer:main',
            'battery_guard = robot_bringup.battery_guard:main',
            'dock = robot_bringup.dock:main',
            'camera_manager = robot_bringup.camera_manager:main',
            'mpu_imu = robot_bringup.mpu_imu:main',
            'service_manager = robot_bringup.service_manager:main',
            'crash_monitor = robot_bringup.crash_monitor:main',
            'battery_standby = robot_bringup.battery_standby:main',
            'pico_io = robot_bringup.pico_io:main',
        ],
    },
)
