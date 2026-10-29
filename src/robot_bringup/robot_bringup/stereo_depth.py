"""Stereo depth from the two IMX219 previews, on demand (VPI stereo on the GPU).

Runs only while something subscribes to one of its outputs (web UI depth view, Foxglove,
`robot depth`, Nav2...), or while force_on is true; otherwise it holds no subscriptions.

  in:  /left/preview/image, /right/preview/image   (640x480, from cameras.launch.py)
  out: /depth/image          sensor_msgs/Image 32FC1, metres, 0 = unknown
       /depth/camera_info    for the rectified left camera
       /depth/points         sensor_msgs/PointCloud2 xyz+rgb (every 4th pixel)
       /depth/preview/compressed  colour-coded depth JPEG (near = red, far = blue)
       TF stereo_camera_left_optical -> stereo_camera_left_rect (static)

Rectification comes from config/calibration/{left,right}.yaml, scaled to the preview size:
the Argus camera node publishes camera_info with K replaced by the rectified P, so its
camera_info can't be used to rectify. The cameras aren't synchronised; frames are paired
within max_pair_gap and the right one is treated as simultaneous.
"""
import os
import time

import cv2
import numpy as np
import rclpy
import vpi
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField
from tf2_ros import StaticTransformBroadcaster

SIDES = ('left', 'right')


def load_calib(path):
    with open(path) as f:
        c = yaml.safe_load(f)
    return {
        'size': (c['image_width'], c['image_height']),
        'K': np.array(c['camera_matrix']['data'], float).reshape(3, 3),
        'D': np.array(c['distortion_coefficients']['data'], float),
        'R': np.array(c['rectification_matrix']['data'], float).reshape(3, 3),
        'P': np.array(c['projection_matrix']['data'], float).reshape(3, 4),
    }


def quat_from_matrix(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    x = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    return x, y, z, w


class StereoDepth(Node):
    def __init__(self):
        super().__init__('stereo_depth')
        self.force_on = self.declare_parameter('force_on', False).value
        self.max_disparity = self.declare_parameter('max_disparity', 64).value   # px at 640 wide: ~0.5 m nearest
        self.max_range = self.declare_parameter('max_range', 6.0).value          # m, beyond this = unknown
        self.max_pair_gap = self.declare_parameter('max_pair_gap', 0.035).value  # s; one frame at 30 fps
        self.speckle_size = self.declare_parameter('speckle_size', 150).value    # px, smaller islands dropped
        calib_dir = os.path.join(get_package_share_directory('robot_bringup'), 'config', 'calibration')
        self.calib = {side: load_calib(os.path.join(calib_dir, f'{side}.yaml')) for side in SIDES}

        self.depth_pub = self.create_publisher(Image, '/depth/image', 5)
        self.info_pub = self.create_publisher(CameraInfo, '/depth/camera_info', 5)
        self.points_pub = self.create_publisher(PointCloud2, '/depth/points', 5)
        self.preview_pub = self.create_publisher(CompressedImage, '/depth/preview/compressed', 5)
        self.static_tf = StaticTransformBroadcaster(self)
        self.publish_rect_frame()

        self.subs, self.frames, self.maps, self.last_done = None, {s: [] for s in SIDES}, None, 0
        self.fps, self.frame_count, self.fps_since = 0.0, 0, time.monotonic()
        self.create_timer(1.0, self.manage)
        self.create_timer(0.03, self.process)

    # ---------- on demand ----------
    def wanted(self):
        self.force_on = self.get_parameter('force_on').value
        return self.force_on or any(p.get_subscription_count() > 0 for p in
                                    (self.depth_pub, self.points_pub, self.preview_pub, self.info_pub))

    def manage(self):
        # Single-threaded executor: creating/destroying here can't race a take on another thread.
        if self.wanted() and self.subs is None:
            self.subs = [self.create_subscription(Image, f'/{side}/preview/image',
                                                  lambda m, s=side: self.on_image(s, m), qos_profile_sensor_data)
                         for side in SIDES]
            self.get_logger().info('depth on (something is watching)')
        elif not self.wanted() and self.subs is not None:
            for sub in self.subs:
                self.destroy_subscription(sub)
            self.subs, self.frames = None, {s: [] for s in SIDES}
            self.get_logger().info('depth off (nobody watching)')
        if self.subs is not None:
            now = time.monotonic()
            self.fps = self.frame_count / (now - self.fps_since)
            self.frame_count, self.fps_since = 0, now

    def on_image(self, side, msg):
        self.frames[side] = (self.frames[side] + [msg])[-6:]

    # ---------- geometry ----------
    def publish_rect_frame(self):
        # Rectified coords = R1 * optical coords, so optical -> rect rotates by R1^T.
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'stereo_camera_left_optical'
        t.child_frame_id = 'stereo_camera_left_rect'
        x, y, z, w = quat_from_matrix(self.calib['left']['R'].T)
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = x, y, z, w
        self.static_tf.sendTransform(t)

    def build_maps(self, size):
        w, h = size
        maps = {}
        for side in SIDES:
            c = self.calib[side]
            S = np.diag([w / c['size'][0], h / c['size'][1], 1.0])  # full-res calibration -> preview pixels
            K, P = S @ c['K'], S @ c['P']
            maps[side] = cv2.initUndistortRectifyMap(K, c['D'], c['R'], P[:, :3], size, cv2.CV_16SC2)
        P1 = np.diag([w / self.calib['left']['size'][0], h / self.calib['left']['size'][1], 1.0]) @ self.calib['left']['P']
        P2 = self.calib['right']['P']
        baseline = -P2[0, 3] / P2[0, 0]
        self.maps, self.size = maps, size
        self.fx, self.fy, self.cx, self.cy = P1[0, 0], P1[1, 1], P1[0, 2], P1[1, 2]
        self.baseline = baseline
        self.get_logger().info(f'rectifying at {w}x{h}: fx {self.fx:.1f}, baseline {baseline * 1000:.1f} mm')

    # ---------- pipeline ----------
    def pair(self):
        best = None
        for a in self.frames['left']:
            ta = Time.from_msg(a.header.stamp).nanoseconds
            if ta <= self.last_done:
                continue
            for b in self.frames['right']:
                gap = abs(ta - Time.from_msg(b.header.stamp).nanoseconds) * 1e-9
                if gap <= self.max_pair_gap and (best is None or ta > best[0]):
                    best = (ta, a, b)
        return best

    @staticmethod
    def gray(msg):
        ch = 3 if msg.encoding in ('rgb8', 'bgr8') else 1
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)[:, :msg.width * ch]
        if ch == 1:
            return img
        img = img.reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if msg.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY)

    def process(self):
        if self.subs is None:
            return
        found = self.pair()
        if found is None:
            return
        stamp_ns, left, right = found
        self.last_done = stamp_ns
        size = (left.width, left.height)
        if self.maps is None or size != self.size:
            self.build_maps(size)
        gl = cv2.remap(self.gray(left), *self.maps['left'], cv2.INTER_LINEAR)
        gr = cv2.remap(self.gray(right), *self.maps['right'], cv2.INTER_LINEAR)
        with vpi.Backend.CUDA:
            vl = vpi.asimage(gl).convert(vpi.Format.Y16_ER, scale=256)
            vr = vpi.asimage(gr).convert(vpi.Format.Y16_ER, scale=256)
            disp = vpi.stereodisp(vl, vr, window=5, maxdisp=self.max_disparity)
            d16 = disp.cpu()  # Q10.5 fixed point (1/32 px)
        # Drop small isolated disparity islands: on plain or shiny floor these are mismatches.
        d16 = np.ascontiguousarray(d16, dtype=np.int16)
        cv2.filterSpeckles(d16, 0, self.speckle_size, 32)  # region < speckle_size px, step > 1 px
        d = d16.astype(np.float32) / 32.0
        depth = np.zeros_like(d, dtype=np.float32)
        # Outside 0.5..max_disparity is VPI's "no match" marker, not a distance.
        valid = (d > 0.5) & (d < self.max_disparity - 0.5)
        depth[valid] = self.fx * self.baseline / d[valid]
        depth[depth > self.max_range] = 0.0
        header = left.header
        header.frame_id = 'stereo_camera_left_rect'
        self.publish(header, depth, gl)
        self.frame_count += 1

    def publish(self, header, depth, gray):
        if self.depth_pub.get_subscription_count() or self.force_on:
            m = Image()
            m.header, m.height, m.width = header, depth.shape[0], depth.shape[1]
            m.encoding, m.step = '32FC1', depth.shape[1] * 4
            m.data = depth.tobytes()
            self.depth_pub.publish(m)
        if self.info_pub.get_subscription_count() or self.depth_pub.get_subscription_count():
            info = CameraInfo()
            info.header, info.width, info.height = header, depth.shape[1], depth.shape[0]
            info.distortion_model, info.d = 'plumb_bob', [0.0] * 5
            info.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
            info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            info.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.info_pub.publish(info)
        if self.points_pub.get_subscription_count():
            self.points_pub.publish(self.cloud(header, depth, gray))
        if self.preview_pub.get_subscription_count():
            vis = np.zeros((*depth.shape, 3), np.uint8)
            near = depth > 0
            # 0.3 m -> red ... max_range -> blue; unknown stays black.
            scaled = np.clip((depth - 0.3) / (self.max_range - 0.3), 0, 1)
            vis[near] = cv2.applyColorMap((255 - scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[near]
            c = CompressedImage()
            c.header, c.format = header, 'jpeg'
            c.data = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
            self.preview_pub.publish(c)

    def cloud(self, header, depth, gray, step=4):
        z = depth[::step, ::step]
        v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
        ok = z > 0
        z, u, v = z[ok], u[ok], v[ok]
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        g = gray[::step, ::step][ok].astype(np.uint32)
        rgb = ((g << 16) | (g << 8) | g).view(np.float32)
        pts = np.stack([x, y, z, rgb], axis=1).astype(np.float32)
        m = PointCloud2()
        m.header, m.height, m.width = header, 1, len(pts)
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                    for i, n in enumerate(('x', 'y', 'z', 'rgb'))]
        m.is_bigendian, m.point_step, m.row_step, m.is_dense = False, 16, 16 * len(pts), True
        m.data = pts.tobytes()
        return m


def main():
    rclpy.init()
    node = StereoDepth()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
