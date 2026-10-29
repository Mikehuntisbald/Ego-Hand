"""Inference-only camera, clock and native MANO contracts. Units are metres."""
from dataclasses import dataclass, field
from pathlib import Path
import re
import numpy as np
from .errors import RuntimeFailure
from .io import read_json

FORBIDDEN = {'gt', 'ground_truth', 'labels', 'qa_score', 'good_point_mask',
             'teacher_prediction', 'T_world_from_camera', 'T_world_from_device'}
ROLES = {'rgb', 'tracking_left', 'tracking_right', 'side_left', 'side_right'}


def ns(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not 0 <= int(value) < 2**63:
        raise RuntimeFailure('timestamp', 'Timestamps must be nonnegative signed-int64 integer ns')
    return int(value)


def reject_privileged(value, trail='input'):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN:
                raise RuntimeFailure('privileged_input', 'Inference input contains a forbidden field', field=trail + '.' + key)
            reject_privileged(item, trail + '.' + key)
    elif isinstance(value, list):
        for item in value:
            reject_privileged(item, trail)


def matrix(value, shape, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != shape or not np.isfinite(result).all():
        raise RuntimeFailure('shape', 'Invalid numeric shape or value', field=name, expected=shape)
    return result


def rotation(value, name='rotation'):
    R = matrix(value, (3, 3), name)
    if np.max(np.abs(R.T @ R - np.eye(3))) > 1e-4 or abs(np.linalg.det(R) - 1) > 1e-4:
        raise RuntimeFailure('rotation', 'Expected a proper SO(3) rotation', field=name)
    return R


def transform(value, name='transform'):
    T = matrix(value, (4, 4), name)
    rotation(T[:3, :3], name)
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-7):
        raise RuntimeFailure('transform', 'Invalid homogeneous transform', field=name)
    return T


@dataclass
class Camera:
    id: str
    role: str
    model: str
    width: int
    height: int
    parameters: list
    T_device_from_camera: np.ndarray
    photometric: str
    input_validated: bool = False

    @classmethod
    def from_dict(cls, item):
        if item['role'] not in ROLES or item['photometric'] not in {'rgb', 'mono'}:
            raise RuntimeFailure('camera_role', 'Unknown camera role or photometric contract')
        if item['model'] not in {'pinhole', 'kb4', 'fisheye624'}:
            raise RuntimeFailure('camera_model', 'Unsupported physical camera model', model=item['model'])
        params = np.asarray(item['parameters'], np.float64)
        expected = {'pinhole': 4, 'kb4': 8, 'fisheye624': 16}[item['model']]
        if params.shape != (expected,) or not np.isfinite(params).all() or min(params[:2]) <= 0:
            raise RuntimeFailure('intrinsics', 'Invalid camera parameters')
        if int(item['width']) <= 1 or int(item['height']) <= 1:
            raise RuntimeFailure('image_size', 'Invalid physical image size')
        return cls(item['id'], item['role'], item['model'], int(item['width']), int(item['height']),
                   params.tolist(), transform(item['T_device_from_camera']), item['photometric'],
                   bool(item.get('input_validated', False)))

    def to_dict(self):
        return dict(self.__dict__)


@dataclass
class Episode:
    episode_id: str
    source_id: str
    source_clock_ns: np.ndarray
    cameras: dict
    frames: list
    source_path: Path
    adapter: str
    provenance: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path):
        path = Path(path).resolve()
        if path.is_dir():
            path = path / 'episode.json'
        raw = read_json(path)
        reject_privileged(raw)
        if raw.get('schema') != 1 or raw.get('units') != 'm':
            raise RuntimeFailure('schema', 'Require episode schema=1 and units=m')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', raw['episode_id']):
            raise RuntimeFailure('episode_id', 'Episode ID must be a safe filename component')
        stamps = np.array([ns(x) for x in raw['source_clock_ns']], dtype=np.int64)
        if len(stamps) == 0 or (np.diff(stamps) <= 0).any() or len(raw['frames']) != len(stamps):
            raise RuntimeFailure('source_clock', 'Source clock must be strictly increasing and match frame count')
        cameras = {x['id']: Camera.from_dict(x) for x in raw['cameras']}
        if len(cameras) != len(raw['cameras']):
            raise RuntimeFailure('camera_id', 'Duplicate physical camera ID')
        roles = [c.role for c in cameras.values()]
        if roles.count('rgb') != 1 or len(set(roles)) != len(roles):
            raise RuntimeFailure('camera_role', 'Exactly one RGB camera and unique roles are required')
        frames = []
        for index, (stamp, frame) in enumerate(zip(stamps, raw['frames'])):
            if ns(frame['timestamp_ns']) != int(stamp) or isinstance(frame['source_index'], bool) or not isinstance(frame['source_index'], int) or frame['source_index'] < 0:
                raise RuntimeFailure('frame_clock', 'Frame/source-clock mismatch')
            images = {}
            for key, image in frame.get('images', {}).items():
                if key not in cameras:
                    raise RuntimeFailure('camera_id', 'Image refers to an unknown camera')
                image = dict(image)
                image['timestamp_ns'] = ns(image['timestamp_ns'])
                if isinstance(image['source_index'], bool) or not isinstance(image['source_index'], int) or image['source_index'] < 0:
                    raise RuntimeFailure('frame_index', 'Camera source index must be an integer')
                location = Path(image['path'])
                image['path'] = str((path.parent / location).resolve() if not location.is_absolute() else location.resolve())
                image['valid'] = bool(image.get('valid', True)) and Path(image['path']).is_file()
                images[key] = image
            frames.append(dict(source_index=int(frame['source_index']), timestamp_ns=int(stamp), images=images))
            if 'cameras' in frame:
                actual = {c['id']: Camera.from_dict(c) for c in frame['cameras']}
                if set(actual)!=set(cameras) or any(actual[k].role!=cameras[k].role for k in cameras):
                    raise RuntimeFailure('frame_calibration', 'Per-frame calibration must preserve physical camera IDs and roles')
                frames[-1]['cameras'] = actual
        return cls(raw['episode_id'], raw.get('source_id', raw['episode_id']), stamps, cameras,
                   frames, path, raw.get('adapter', 'manifest'), raw.get('provenance', {}))

    def camera_for_role(self, role):
        return next((c for c in self.cameras.values() if c.role == role), None)

    def camera_at(self, camera_id, timestamp_ns):
        i = int(np.searchsorted(self.source_clock_ns, timestamp_ns))
        options = [k for k in [i-1,i] if 0<=k<len(self.frames)]
        index = min(options, key=lambda j: abs(int(self.source_clock_ns[j])-int(timestamp_ns)))
        return self.frames[index].get('cameras', self.cameras)[camera_id]

    def capabilities(self):
        roles = {c.role: c for c in self.cameras.values()}
        return dict(stereo_vo=all(k in roles for k in ['tracking_left', 'tracking_right']),
                    physical_stereo_depth=any(k in roles for k in ['tracking_left', 'tracking_right']),
                    side_rgb_recovery=any(c.role.startswith('side_') and c.photometric == 'rgb' and c.input_validated
                                          for c in self.cameras.values()),
                    IMU_fusion=False, dataset=self.adapter,
                    mego_formal_export=False)

    def interval(self, start_ns=None, end_ns=None):
        start = 0 if start_ns is None else ns(start_ns)
        end = 2**63 - 1 if end_ns is None else ns(end_ns)
        if start > end:
            raise RuntimeFailure('interval', 'Start is after end')
        return [i for i, t in enumerate(self.source_clock_ns) if start <= int(t) <= end]
