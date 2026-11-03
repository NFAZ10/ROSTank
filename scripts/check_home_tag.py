#!/usr/bin/env python3
"""Stationary AprilTag check. Never publishes velocity or navigation goals."""
import argparse
from pathlib import Path
import time

import cv2
import numpy as np
import yaml


def detect(gray, tag_id, tag_size, calibration):
    detector = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11))
    corners, ids, _ = detector.detectMarkers(gray)
    seen = [] if ids is None else ids.ravel().tolist()
    if tag_id not in seen:
        return seen, None
    c = calibration
    scale = np.diag([gray.shape[1] / c['image_width'],
                     gray.shape[0] / c['image_height'], 1.0])
    K = scale @ np.array(c['camera_matrix']['data'], float).reshape(3, 3)
    D = np.array(c['distortion_coefficients']['data'], float)
    h = tag_size / 2
    points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float32)
    ok, _, translation = cv2.solvePnP(
        points, corners[seen.index(tag_id)].reshape(4, 2), K, D,
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok or not np.isfinite(translation).all() or translation[2, 0] <= 0:
        return seen, None
    return seen, translation.ravel()


def image_gray(msg):
    channels = {'rgb8': 3, 'bgr8': 3, 'mono8': 1}.get(msg.encoding)
    if channels is None:
        raise ValueError(f'Unsupported camera encoding: {msg.encoding}')
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    pixels = rows[:, :msg.width * channels]
    if channels == 1:
        return pixels
    pixels = pixels.reshape(msg.height, msg.width, channels)
    code = cv2.COLOR_RGB2GRAY if msg.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(pixels, code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag-id', type=int, default=0)
    parser.add_argument('--tag-size', type=float, default=0.10,
                        help='Measured outer black square width in metres')
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--image', help='Check a saved image instead of live ROS')
    parser.add_argument('--calibration', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'src/robot_bringup/config/calibration/left.yaml')
    args = parser.parse_args()
    if not np.isfinite(args.tag_size) or args.tag_size <= 0:
        parser.error('--tag-size must be positive and finite')
    if not np.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('--timeout must be positive and finite')
    with args.calibration.open() as stream:
        calibration = yaml.safe_load(stream)

    def report(gray):
        seen, translation = detect(gray, args.tag_id, args.tag_size, calibration)
        if translation is not None:
            x, y, z = translation
            print(f'Home tag {args.tag_id} detected: {np.linalg.norm(translation):.2f} m range; '
                  f'camera coordinates right={x:.2f}, down={y:.2f}, forward={z:.2f} m.')
            return True
        if seen:
            print(f'Tag IDs visible: {seen}; waiting for home ID {args.tag_id}.')
        return False

    if args.image:
        gray = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            parser.error('Cannot read --image')
        return 0 if report(gray) else 1

    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node('check_home_tag')
    state = {'frames': 0, 'found': False}

    def receive(msg):
        state['frames'] += 1
        state['found'] = report(image_gray(msg))

    node.create_subscription(Image, '/left/preview/image', receive, qos_profile_sensor_data)
    print(f'Watching left camera for tag36h11 ID {args.tag_id} ({args.tag_size:g} m). '
          'No movement commands are sent.')
    deadline = time.monotonic() + args.timeout
    try:
        while rclpy.ok() and not state['found'] and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        if not state['found']:
            print('No camera frames received: check robot stack and ROS_DOMAIN_ID.'
                  if not state['frames'] else
                  'Home tag not detected: check visibility, lighting, family, and ID.')
        return 0 if state['found'] else 1
    except KeyboardInterrupt:
        return 130
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
