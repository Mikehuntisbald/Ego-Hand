"""Physical camera geometry. No dataset reference pose is consumed."""
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, '/mnt/why/HOT3D-hand-tracking-toolkit')
from hand_tracking_toolkit.camera import from_json


IDENTITY = dict(quaternion_wxyz=[1., 0., 0., 0.], translation_xyz=[0., 0., 0.])


def transform(pose):
    q = pose['quaternion_wxyz']
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(q[1:]+q[:1]).as_matrix()
    result[:3, 3] = pose['translation_xyz']
    return result


def camera(calibration):
    # The toolkit asks for a world pose even for local project/unproject.
    # Identity here is a local coordinate definition, not a reference trajectory.
    return from_json(dict(calibration=calibration, T_world_from_camera=IDENTITY))


def remap_geometry(calibration, width, height, focal, principal=None, native_from_virtual=None):
    principal = np.array(principal if principal is not None else [(width-1)/2, (height-1)/2])
    yy, xx = np.mgrid[:height, :width]
    directions = np.stack([(xx-principal[0])/focal, (yy-principal[1])/focal, np.ones_like(xx)], -1)
    if native_from_virtual is not None:
        directions = directions @ np.asarray(native_from_virtual).T
    native = camera(calibration)
    uv = native.eye_to_window(directions).astype(np.float32)
    angle = np.arccos(np.clip(directions[..., 2]/np.linalg.norm(directions, axis=-1), -1, 1))
    valid = (np.isfinite(uv).all(-1) & (uv[..., 0] >= 0) & (uv[..., 0] < native.width-1) &
             (uv[..., 1] >= 0) & (uv[..., 1] < native.height-1) &
             (angle < calibration.get('max_solid_angle', 1.4)))
    uv[~valid] = -1
    return uv, valid


def remap_image(image, uv):
    return cv2.remap(image, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def rectified_pair(cameras, first, second, width=480, height=288, focal=200.):
    transforms = [transform(cameras[key]['calibration']['T_device_from_camera']) for key in [first, second]]
    baseline_direction = transforms[1][:3, 3]-transforms[0][:3, 3]
    baseline = np.linalg.norm(baseline_direction)
    assert baseline > .001
    x = baseline_direction/baseline
    z = transforms[0][:3, 2]+transforms[1][:3, 2]
    z -= x*np.dot(x, z)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    device_from_rectified = np.stack([x, y, z], 1)
    maps, validity = [], []
    for key, native_transform in zip([first, second], transforms):
        rotation = native_transform[:3, :3].T @ device_from_rectified
        uv, valid = remap_geometry(cameras[key]['calibration'], width, height, focal,
                                  native_from_virtual=rotation)
        maps.append(uv)
        validity.append(valid)
    return dict(maps=maps, validity=validity, baseline_m=float(baseline), focal_px=float(focal),
                principal=[(width-1)/2, (height-1)/2], device_from_rectified=device_from_rectified,
                source_camera=first, target_camera=second, first_device_transform=transforms[0])


def load_frames(folder):
    folder = Path(folder)
    frames = []
    for path in sorted(folder.glob('*.info.json')):
        info = json.loads(path.read_text())
        prefix = path.name.split('.')[0]
        calibration = json.loads((folder/f'{prefix}.cameras.json').read_text())
        assert all('T_world_from_camera' not in item for item in calibration.values()), 'Use sanitized inference calibration'
        frames.append(dict(id=prefix, timestamp_ns=int(info['ref_timestamp_ns']),
                           image_timestamps_ns=info['image_timestamps_ns'], cameras=calibration,
                           images={key: str(folder/f'{prefix}.image_{key}.jpg') for key in calibration}))
    times = np.array([row['timestamp_ns'] for row in frames], dtype=np.int64)
    assert len(times) > 1 and (np.diff(times) > 0).all()
    return frames
