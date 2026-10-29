"""Prepare canonical inference metadata from extracted HOT3D Aria frames.

Only physical calibration is projected into the output. Hand labels and hand-shape
files are never opened. The prepare step is separate from inference and scoring.
"""
from pathlib import Path
import numpy as np
from ..errors import RuntimeFailure
from ..io import read_json, atomic_json, sha256
from ..schema import ns


def pose_from_json(item):
    from ..geometry import quaternion_matrix
    T = np.eye(4)
    T[:3, :3] = quaternion_matrix(item['quaternion_wxyz'], order='wxyz')
    T[:3, 3] = item['translation_xyz']
    return T


def calibration(item, camera_id, role):
    data = item.get('calibration', item)
    projection = data.get('projection', {})
    model = str(data.get('projection_model_type', projection.get('model', projection.get('name', '')))).lower()
    values = data.get('projection_params', projection.get('params'))
    if values is None:
        raise RuntimeFailure('hot3d_calibration', 'Missing projection parameters', camera=camera_id)
    p = list(values)
    if '624' in model:
        # HOT3D's FISHEYE624 packs a shared focal followed by cx,cy and 12 distortion terms.
        if len(p) == 15:
            p = [p[0], p[0], *p[1:]]
        name = 'fisheye624'
    elif 'kannala' in model or 'kb4' in model:
        name = 'kb4'
    elif 'pinhole' in model:
        name = 'pinhole'
    else:
        raise RuntimeFailure('hot3d_calibration', 'Unsupported camera projection', camera=camera_id, model=model)
    size = data.get('image_size', [data.get('image_width'), data.get('image_height')])
    extrinsic = data.get('T_device_from_camera')
    if not extrinsic:
        raise RuntimeFailure('hot3d_calibration', 'Physical extrinsic is required', camera=camera_id)
    return dict(id=camera_id, role=role, model=name, width=int(size[0]), height=int(size[1]),
                parameters=p, T_device_from_camera=pose_from_json(extrinsic),
                photometric='rgb' if role == 'rgb' else 'mono', input_validated=False)


def prepare(folder, output, episode_id=None):
    folder, output = Path(folder).resolve(), Path(output).resolve()
    if output.exists():
        raise RuntimeFailure('exists', 'Preparation refuses to overwrite an existing manifest')
    paths = sorted(folder.glob('*.info.json'))
    if not paths:
        raise RuntimeFailure('hot3d_input', 'No extracted HOT3D info files found')
    roles = {'214-1': 'rgb', '1201-1': 'tracking_left', '1201-2': 'tracking_right'}
    cameras, frames, discarded = None, [], 0
    for source_index, path in enumerate(paths):
        prefix = path.name.removesuffix('.info.json')
        info = read_json(path)
        physical = read_json(folder / (prefix + '.cameras.json'))
        discarded += sum('T_world_from_camera' in x for x in physical.values())
        current = [calibration(physical[key], key, role) for key, role in roles.items() if key in physical]
        if cameras is None:
            cameras = current
        images = {}
        for c in cameras:
            key = c['id']
            name = folder / (prefix + '.image_' + key + '.jpg')
            if key in info.get('image_timestamps_ns', {}):
                images[key] = dict(path=str(name), source_index=int(prefix) if prefix.isdigit() else source_index,
                                   timestamp_ns=ns(info['image_timestamps_ns'][key]), valid=name.is_file())
        frames.append(dict(source_index=int(prefix) if prefix.isdigit() else source_index,
                           timestamp_ns=ns(info['ref_timestamp_ns']), images=images,
                           cameras=current))
    result = dict(schema=1, episode_id=episode_id or folder.name, source_id=folder.name,
                  units='m', adapter='hot3d_aria', cameras=cameras, frames=frames,
                  source_clock_ns=[f['timestamp_ns'] for f in frames],
                  provenance=dict(preparation='physical calibration projection only',
                                  discarded_reference_pose_fields=discarded, hand_labels_opened=False))
    atomic_json(output, result)
    return result
