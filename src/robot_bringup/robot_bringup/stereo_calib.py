"""Stereo checkerboard calibration for the two IMX219s (used by web_ui's /calibrate page).

Board detection, sample bookkeeping, coverage, the OpenCV calibration and the
camera_info YAML output; no ROS here. Coverage and corner ordering follow ROS
camera_calibration, so results compare with cameracalibrator.

Samples (full-resolution corners, plus JPEGs for the record) are kept in
~/robot_ws/calibration_captures/<session>/ so a calibration can be recomputed later.
"""
import glob
import json
import math
import os
import re
import shutil
import time

import cv2
import numpy as np

SIDES = ('left', 'right')
# A parameter (x, y, size, skew) counts as fully covered once samples span this much
# of it (ROS camera_calibration uses 0.4 for size; a small board can't reach that
# without leaving MIN_SIZE, so 0.25 here).
PARAM_RANGES = (0.7, 0.7, 0.25, 0.5)
MIN_SAMPLES = 10
GOOD_SAMPLES = 30
NOVELTY = 0.2      # a capture must differ from every kept sample by this much (sum of params)
MIN_SIZE = 0.12    # board smaller than this in the image: its pose is too poorly measured
MAX_MOTION_PX = 0.5  # corner shift allowed between two frames ~0.1 s apart (cameras aren't synced)
# A rigid side-by-side board: anything beyond these is a bad calibration, not the hardware.
MAX_OFFSET_MM = (4.0, 5.0)  # vertical, depth
MAX_ROTATION_DEG = 3.0
EDGE_PX = 15        # corners closer than this to the image border: board not fully in view
MAX_TILT_DEG = 55   # steeper boards gave badly placed corners (session 2026-10-28, samples 33/36/37)
# Grades (reprojection error in pixels at 1640x1232): (per camera, stereo).
GRADES = (('excellent', 0.5, 0.6), ('good', 0.8, 1.0))


def to_gray(img, encoding):
    if encoding in ('rgb8', 'rgba8'):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY if encoding == 'rgb8' else cv2.COLOR_RGBA2GRAY)
    if encoding in ('bgr8', 'bgra8'):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY if encoding == 'bgr8' else cv2.COLOR_BGRA2GRAY)
    if encoding == 'mono8':
        return img
    raise ValueError(f'unsupported image encoding {encoding}')


def canonical_order(corners, cols, rows):
    """Same first corner in both cameras (ROS camera_calibration's rule): an even x even
    board is symmetric, so OpenCV may start from opposite corners in the two views."""
    c = corners.reshape(-1, 2)
    if cols != rows:
        if c[0, 1] > c[-1, 1]:
            c = c[::-1]
    else:
        d = (c[-1] - c[0]) >= 0
        if not d.all():
            if not d.any():
                c = c[::-1]
            elif d[0]:
                c = np.rot90(c.reshape(rows, cols, 2)).reshape(-1, 2)
            else:
                c = np.rot90(c.reshape(rows, cols, 2), 3).reshape(-1, 2)
    return np.ascontiguousarray(c, dtype=np.float32)


def find_corners(gray, pattern, accurate):
    """pattern = (cols, rows) inner corners. accurate: full-res capture; else fast preview check."""
    cols, rows = pattern
    if accurate:
        ok, c = cv2.findChessboardCornersSB(
            gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok:  # SB wants a clean border; the classic detector is more forgiving
            ok, c = cv2.findChessboardCorners(
                gray, pattern, flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ok:
                c = cv2.cornerSubPix(gray, c, (11, 11), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
    else:
        ok, c = cv2.findChessboardCorners(
            gray, pattern,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK)
    return canonical_order(c, cols, rows) if ok else None


def sample_params(c, cols, size):
    """x, y, size, skew of one board view in 0..1 (ROS camera_calibration's measures)."""
    w, h = size
    ul, ur, dr, dl = c[0], c[cols - 1], c[-1], c[-cols]
    a, b = ur - ul, dl - ul
    p, q = dr - ur, dl - dr
    area = abs(a[0] * b[1] - a[1] * b[0]) / 2.0 + abs(p[0] * q[1] - p[1] * q[0]) / 2.0
    border = math.sqrt(area)
    mx, my = c.mean(axis=0)
    px = min(1.0, max(0.0, (mx - border / 2) / max(1.0, w - border)))
    py = min(1.0, max(0.0, (my - border / 2) / max(1.0, h - border)))
    v1, v2 = ul - ur, dr - ur
    ang = math.acos(max(-1.0, min(1.0, float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))))
    skew = min(1.0, 2.0 * abs(math.pi / 2 - ang))
    return [px, py, math.sqrt(area / (w * h)), skew]


def coverage(params):
    if not params:
        return [0.0] * 4
    arr = np.array(params)
    return [round(min(1.0, (arr[:, i].max() - arr[:, i].min()) / PARAM_RANGES[i]), 2) for i in range(4)]


def camera_info_yaml(name, size, K, D, R, P):
    fmt = lambda a: '[' + ', '.join(f'{v:.9g}' for v in np.asarray(a).ravel()) + ']'
    return (f'# Stereo calibration from the web UI (/calibrate), {time.strftime("%Y-%m-%d %H:%M")}.\n'
            f'image_width: {size[0]}\nimage_height: {size[1]}\ncamera_name: {name}\n'
            f'camera_matrix: {{rows: 3, cols: 3, data: {fmt(K)}}}\n'
            f'distortion_model: plumb_bob\n'
            f'distortion_coefficients: {{rows: 1, cols: 5, data: {fmt(np.asarray(D).ravel()[:5])}}}\n'
            f'rectification_matrix: {{rows: 3, cols: 3, data: {fmt(R)}}}\n'
            f'projection_matrix: {{rows: 3, cols: 4, data: {fmt(P)}}}\n')


class StereoSession:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.config_path = os.path.join(root, 'config.json')
        self.cols, self.rows, self.square_mm = 6, 4, 21.0
        try:
            with open(self.config_path) as f:
                cfg = json.load(f)
            self.cols, self.rows, self.square_mm = int(cfg['cols']), int(cfg['rows']), float(cfg['square_mm'])
        except (OSError, ValueError, KeyError):
            pass
        self.reset()

    # ---------- setup ----------
    @property
    def pattern(self):
        return (self.cols, self.rows)

    def configure(self, cols, rows, square_mm):
        cols, rows, square_mm = int(cols), int(rows), float(square_mm)
        if not (2 <= cols <= 30 and 2 <= rows <= 30 and 3 <= square_mm <= 200):
            raise ValueError('inner corners 2-30 each way, squares 3-200 mm')
        changed = (cols, rows) != self.pattern or square_mm != self.square_mm
        self.cols, self.rows, self.square_mm = cols, rows, square_mm
        with open(self.config_path, 'w') as f:
            json.dump({'cols': cols, 'rows': rows, 'square_mm': square_mm}, f)
        if changed:
            self.reset()

    def reset(self):
        self.samples = []      # {'left': corners, 'right': corners, 'params': {side: [4]}}
        self.size = None       # full-res (w, h)
        self.result = None
        self.last_pair = None  # gray images of the newest capture, for the rectified check
        self.dir = os.path.join(self.root, time.strftime('session_%Y%m%d_%H%M%S'))

    # ---------- samples ----------
    def novelty(self, params):
        """Smallest difference to a kept sample (left camera's params); inf if none."""
        if not self.samples:
            return math.inf
        return min(sum(abs(a - b) for a, b in zip(params, s['params']['left'])) for s in self.samples)

    def add(self, grays, colors, left_next=None):
        """grays/colors: {side: full-res image}; left_next: a later left frame to check the
        board was still. Returns a message; raises if the sample isn't usable."""
        size = grays['left'].shape[1], grays['left'].shape[0]
        if self.size and size != self.size:
            raise RuntimeError('image size changed: clear the samples and start again')
        corners = {}
        for side in SIDES:
            c = find_corners(grays[side], self.pattern, accurate=True)
            if c is None:
                raise RuntimeError(f"the {side} camera lost the board at full resolution: hold it still")
            corners[side] = c
        w, h = size
        for side in SIDES:
            c = corners[side]
            if (c[:, 0].min() < EDGE_PX or c[:, 1].min() < EDGE_PX
                    or c[:, 0].max() > w - EDGE_PX or c[:, 1].max() > h - EDGE_PX):
                raise RuntimeError(f'the board is at the edge of the {side} view: move it fully into view')
        tilt = self.tilt_deg(corners['left'], size)
        if tilt > MAX_TILT_DEG:
            raise RuntimeError(f'tilted too far ({tilt:.0f}°): about 30–45° is enough')
        params = {side: sample_params(corners[side], self.cols, size) for side in SIDES}
        if min(params['left'][2], params['right'][2]) < MIN_SIZE:
            raise RuntimeError('board too far away to measure well: bring it closer')
        if self.novelty(params['left']) < NOVELTY:
            raise RuntimeError('too similar to a sample already taken: move or tilt the board')
        if left_next is not None:
            c2 = find_corners(left_next, self.pattern, accurate=True)
            moved = math.inf if c2 is None else float(np.linalg.norm(c2 - corners['left'], axis=1).mean())
            if moved > MAX_MOTION_PX:
                raise RuntimeError(f'the board moved during the capture ({moved:.1f} px): '
                                   'prop it up or hold it stiller')
        self.size = size
        self.samples.append({'left': corners['left'], 'right': corners['right'], 'params': params})
        self.last_pair = grays
        self.result = None
        os.makedirs(self.dir, exist_ok=True)
        n = len(self.samples)
        for side in SIDES:
            cv2.imwrite(os.path.join(self.dir, f'{side}_{n:02d}.jpg'), colors[side], [cv2.IMWRITE_JPEG_QUALITY, 92])
        self.save_corners()
        return f'sample {n} captured'

    def board_points(self):
        obj = np.zeros((self.rows * self.cols, 3), np.float32)
        obj[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2) * (self.square_mm / 1000.0)
        return obj

    def tilt_deg(self, corners, size):
        """Rough angle between the board and the image plane (rough intrinsics: a coarse gate)."""
        w, h = size
        f = 1100.0 * w / 1640.0  # IMX219 at this mode, from the calibrations so far
        K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        ok, rv, _ = cv2.solvePnP(self.board_points(), corners, K, None)
        if not ok:
            return 0.0
        normal_z = cv2.Rodrigues(rv)[0][2, 2]
        return float(np.degrees(np.arccos(min(1.0, abs(normal_z)))))

    def load(self, session_dir):
        """Reload a saved session (samples.json) to recalibrate without recapturing."""
        with open(os.path.join(session_dir, 'samples.json')) as f:
            d = json.load(f)
        self.cols, self.rows, self.square_mm = int(d['cols']), int(d['rows']), float(d['square_mm'])
        with open(self.config_path, 'w') as f:
            json.dump({'cols': self.cols, 'rows': self.rows, 'square_mm': self.square_mm}, f)
        self.reset()
        self.dir, self.size = session_dir, tuple(d['size'])
        for smp in d['samples']:
            c = {side: np.array(smp[side], np.float32) for side in SIDES}
            self.samples.append({**c, 'params': {side: sample_params(c[side], self.cols, self.size) for side in SIDES}})
        n = len(self.samples)
        pair = {side: cv2.imread(os.path.join(session_dir, f'{side}_{n:02d}.jpg'), cv2.IMREAD_GRAYSCALE)
                for side in SIDES}
        self.last_pair = pair if all(v is not None for v in pair.values()) else None
        return f'loaded {n} samples from {os.path.basename(session_dir)}'

    @staticmethod
    def sessions(root):
        out = []
        for d in sorted(glob.glob(os.path.join(root, 'session_*')), reverse=True):
            try:
                with open(os.path.join(d, 'samples.json')) as f:
                    n = len(json.load(f)['samples'])
            except (OSError, ValueError, KeyError):
                continue
            out.append({'name': os.path.basename(d), 'samples': n})
        return out

    def undo(self):
        if self.samples:
            self.samples.pop()
            self.result = None
            self.save_corners()

    def save_corners(self):
        if not self.samples:
            return
        with open(os.path.join(self.dir, 'samples.json'), 'w') as f:
            json.dump({'cols': self.cols, 'rows': self.rows, 'square_mm': self.square_mm, 'size': self.size,
                       'samples': [{s: self.samples[i][s].tolist() for s in SIDES}
                                   for i in range(len(self.samples))]}, f)

    def progress(self):
        return {side: coverage([s['params'][side] for s in self.samples]) for side in SIDES}

    # ---------- calibration ----------
    def compute(self):
        if len(self.samples) < MIN_SAMPLES:
            raise RuntimeError(f'need at least {MIN_SAMPLES} samples (have {len(self.samples)})')
        obj = self.board_points()
        keep = list(range(len(self.samples)))
        budget = len(self.samples) // 4  # never drop more than a quarter
        dropped = []                     # [sample number, reason]
        result = self._solve(obj, keep)

        def drop(bad):
            nonlocal keep, result
            if not bad or len(dropped) + len(bad) > budget or len(keep) - len(bad) < MIN_SAMPLES:
                return False
            dropped.extend([i + 1, why] for i, why in sorted(bad.items()))
            keep = [i for i in keep if i not in bad]
            result = self._solve(obj, keep)
            return True

        # 1. One camera's own corners are off (steep tilt, blur, misdetection).
        for _ in range(3):
            bad = {}
            for side in SIDES:
                errs = result['mono'][side]['per_sample']
                limit = max(1.0, 3.0 * float(np.median(errs)))
                for j, e in enumerate(errs):
                    if e > limit:
                        bad[keep[j]] = f'{side} camera {e:.1f} px'
            if not drop(bad):
                break
        # 2. Left and right views disagree (board moved between the two exposures).
        errs = result['stereo_err']
        limit = max(1.0, 3.0 * float(np.median(errs)))
        drop({keep[j]: f'left/right disagree {e:.1f} px' for j, e in enumerate(errs) if e > limit})
        self.result = result
        R, T = result['R'], result['T']
        t_mm = T.ravel() * 1000
        rot = float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
        warnings = []
        if abs(t_mm[1]) > MAX_OFFSET_MM[0]:
            warnings.append(f'cameras come out {abs(t_mm[1]):.1f} mm apart vertically')
        if abs(t_mm[2]) > MAX_OFFSET_MM[1]:
            warnings.append(f'cameras come out {abs(t_mm[2]):.1f} mm apart front-to-back')
        if rot > MAX_ROTATION_DEG:
            warnings.append(f'cameras come out rotated {rot:.1f}° relative to each other')
        order = sorted(range(len(keep)), key=lambda j: -result['stereo_err'][j])
        result['summary'] = {
            'samples': len(keep), 'dropped': dropped,
            'rms_left': round(result['mono']['left']['rms'], 3), 'rms_right': round(result['mono']['right']['rms'], 3),
            'rms_stereo': round(result['rms_stereo'], 3),
            'baseline_mm': round(float(np.linalg.norm(T)) * 1000, 1),
            'offset_mm': [round(float(v), 1) for v in t_mm],
            'fx_left': round(float(result['mono']['left']['K'][0, 0]), 1),
            'fx_right': round(float(result['mono']['right']['K'][0, 0]), 1),
            'rotation_deg': round(rot, 2),
            'worst_samples': [keep[j] + 1 for j in order[:3]],
            'warnings': warnings,
        }
        mono_rms = max(result['mono']['left']['rms'], result['mono']['right']['rms'])
        grade = next((name for name, m, st in GRADES if mono_rms < m and result['rms_stereo'] < st), 'poor')
        result['summary']['grade'] = grade
        result['summary']['good'] = grade != 'poor' and not warnings
        return result['summary']

    def _solve(self, obj, keep):
        """Mono + stereo calibration on the samples in keep; per-sample stereo errors."""
        objs = [obj] * len(keep)
        pts = {side: [self.samples[i][side].reshape(-1, 1, 2) for i in keep] for side in SIDES}
        mono = {}
        for side in SIDES:
            rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objs, pts[side], self.size, None, None)
            per = []
            for o, p, rv, tv in zip(objs, pts[side], rvecs, tvecs):
                proj, _ = cv2.projectPoints(o, rv, tv, K, D)
                per.append(float(np.sqrt(np.mean(np.sum((proj - p) ** 2, axis=2)))))
            mono[side] = {'rms': rms, 'K': K, 'D': D, 'per_sample': per}
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        rms_s, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
            objs, pts['left'], pts['right'], mono['left']['K'], mono['left']['D'],
            mono['right']['K'], mono['right']['D'], self.size, criteria=crit, flags=cv2.CALIB_FIX_INTRINSIC)
        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            mono['left']['K'], mono['left']['D'], mono['right']['K'], mono['right']['D'], self.size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
        # Stereo error per sample: board pose from the left view, carried into the right
        # camera through R, T, compared with what the right camera saw.
        stereo_err = []
        for o, pl, pr in zip(objs, pts['left'], pts['right']):
            _, rv, tv = cv2.solvePnP(o, pl, mono['left']['K'], mono['left']['D'])
            Rl = cv2.Rodrigues(rv)[0]
            proj, _ = cv2.projectPoints(o, cv2.Rodrigues(R @ Rl)[0], R @ tv + T,
                                        mono['right']['K'], mono['right']['D'])
            stereo_err.append(float(np.sqrt(np.mean(np.sum((proj - pr) ** 2, axis=2)))))
        return {'mono': mono, 'R': R, 'T': T, 'R1': R1, 'R2': R2, 'P1': P1, 'P2': P2,
                'rms_stereo': rms_s, 'stereo_err': stereo_err}

    def rectified_jpeg(self, pair=None):
        """Side-by-side rectified view with horizontal guide lines: features must sit on the same row."""
        r = self.result
        pair = pair or self.last_pair
        if r is None or pair is None:
            return None
        out = []
        for side, Rr, P in (('left', r['R1'], r['P1']), ('right', r['R2'], r['P2'])):
            m1, m2 = cv2.initUndistortRectifyMap(r['mono'][side]['K'], r['mono'][side]['D'], Rr, P[:, :3],
                                                 self.size, cv2.CV_16SC2)
            out.append(cv2.cvtColor(cv2.remap(pair[side], m1, m2, cv2.INTER_LINEAR), cv2.COLOR_GRAY2BGR))
        img = np.hstack(out)
        for y in range(0, img.shape[0], 48):
            cv2.line(img, (0, y), (img.shape[1], y), (0, 200, 0), 1)
        img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
        return cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def write_yaml(self, calib_dir, urdf_path=None):
        """Back up the current files, write left.yaml / right.yaml, update the URDF baseline."""
        r = self.result
        if r is None:
            raise RuntimeError('calibrate first')
        backup = os.path.join(calib_dir, time.strftime('backup_%Y%m%d_%H%M%S'))
        os.makedirs(backup, exist_ok=True)
        for side in SIDES:
            path = os.path.join(calib_dir, f'{side}.yaml')
            if os.path.exists(path):
                shutil.copy2(path, backup)
        for side, Rr, P in (('left', r['R1'], r['P1']), ('right', r['R2'], r['P2'])):
            with open(os.path.join(calib_dir, f'{side}.yaml'), 'w') as f:
                f.write(camera_info_yaml(side, self.size, r['mono'][side]['K'], r['mono'][side]['D'], Rr, P))
        if urdf_path and os.path.exists(urdf_path):
            baseline = float(np.linalg.norm(r['T']))
            with open(urdf_path) as f:
                text = f.read()
            new = re.sub(r'(<xacro:property name="camera_baseline" value=")[0-9.]+(")',
                         lambda m: f'{m.group(1)}{baseline:.4f}{m.group(2)}', text)
            if new != text:
                with open(urdf_path, 'w') as f:
                    f.write(new)
        return backup

