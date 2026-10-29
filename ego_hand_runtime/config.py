"""Explicit profiles; no implicit environment switches or fixed GPU ownership."""
import copy
import os
from pathlib import Path
from .errors import RuntimeFailure
from .io import read_json

DEFAULT = {
    'schema': 1, 'device': 'cuda:0', 'backend': 'live', 'environment': {},
    'assets': {}, 'sources': {}, 'dit': {'enabled': False},
    'replay': {'bundle': None},
    'sampling': {'primary_fps': None, 'stereo_stride': 2, 'max_skew_ns': 5000000},
    'pools': {'rf': 1, 'wilor': 1, 'unidepth': 1, 'headpose': 1, 'stereo': 1, 'ba': 1},
    'scheduler': {'workers': 1, 'queue_capacity': 8, 'admission_timeout_s': 60},
    'resources': {'gpu_admit_gib': 8.0, 'gpu_resume_gib': 7.0, 'gpu_hard_gib': 10.0,
                  'ram_admit_gib': 12.0, 'ram_resume_gib': 10.0, 'ram_hard_gib': 18.0},
    'decode': {'primary': 'cpu', 'tracking': 'cpu', 'stereo': 'cpu', 'window_frames': 4},
    'headpose': {'backend': 'cuvslam', 'expected_sdk_sha256': None, 'max_gap_ns': 100000000,
                 'max_rig_drift_m': 0.0001, 'max_rig_drift_rad': 0.001,
                 'max_speed_m_s': 10.0, 'max_angular_speed_rad_s': 30.0, 'orb_fallback': False},
    'depth': {'backend': 'lightess', 'mono': 'disabled', 'confidence_min': 0.2,
              'focal_px': 200.0, 'width': 480, 'height': 288, 'min_samples': 6},
    'ba': {'iterations': 100, 'loss_profile': 'frozen_native_v1', 'mask_outside_image': False,
           'world_root_so3_weight': 0.0, 'wrist_motion_weight': 0.0,
           'observation_preservation_weight': 0.0, 'max_gap_ns': 100000000},
    'quality': {'enabled': True, 'max_gap_ns': 100000000, 'min_side_confidence': 0.5,
                'rmse_m': 0.005, 'p95_m': 0.010, 'maximum_m': 0.030,
                'reprojection_increase_px': 0.2},
    'delivery': {'format': 'aria', 'max_interpolation_gap_ns': 100000000},
}


def merge(base, update, prefix='config'):
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key not in base and prefix not in {'config.assets', 'config.sources', 'config.environment'}:
            raise RuntimeFailure('config_key', 'Unknown configuration key', field=prefix + '.' + key)
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            result[key] = merge(base[key], value, prefix + '.' + key)
        else:
            result[key] = value
    return result


def load_config(path=None, device=None):
    cfg = merge(DEFAULT, read_json(path) if path else {})
    if device:
        cfg['device'] = device
    if cfg['schema'] != 1 or cfg['backend'] not in {'live', 'replay'}:
        raise RuntimeFailure('config', 'Unsupported schema or backend')
    if cfg['depth']['mono'] not in {'disabled', 'torch'} or cfg['depth']['backend']!='lightess':
        raise RuntimeFailure('unsupported', 'Use the verified LightESS ABI; mono supports explicit disabled or Torch')
    if cfg['ba']['loss_profile']!='frozen_native_v1':
        raise RuntimeFailure('unsupported', 'Unknown native loss normalization')
    for key in ['workers', 'queue_capacity']:
        value = cfg['scheduler'][key]
        if isinstance(value, bool) or not isinstance(value, int) or value<1:
            raise RuntimeFailure('scheduler', 'Worker and queue counts must be positive integers')
    for value in [cfg['sampling']['stereo_stride'], cfg['decode']['window_frames'], cfg['ba']['iterations']]:
        if isinstance(value, bool) or not isinstance(value, int) or value<1:
            raise RuntimeFailure('positive_integer', 'Stride, decoder window and BA iterations must be positive integers')
    for key in ['world_root_so3_weight', 'wrist_motion_weight', 'observation_preservation_weight']:
        if not 0<=cfg['ba'][key]<float('inf'):
            raise RuntimeFailure('loss_weight', 'Loss weights must be finite and nonnegative')
    if cfg['headpose']['backend'] != 'cuvslam':
        raise RuntimeFailure('headpose', 'Normal headpose must use stereo cuVSLAM VO')
    if cfg['headpose']['orb_fallback']:
        raise RuntimeFailure('unsupported', 'CPU ORB fallback is not integrated in this public revision')
    if cfg['dit']['enabled']:
        raise RuntimeFailure('unsupported', 'DiT remains an unadmitted research profile; use the frozen isolated research interface')
    if cfg['delivery']['format'] not in {'aria', 'internal'}:
        raise RuntimeFailure('unsupported', 'No validated MeGo data/export contract is available')
    if cfg['decode']['primary'] != 'cpu' or cfg['decode']['tracking'] != 'cpu' or cfg['decode']['stereo'] != 'cpu':
        raise RuntimeFailure('unsupported', 'NVDEC is not validated in this runtime; do not label CPU fallback as GPU decoding')
    for key in ['rf', 'wilor', 'unidepth', 'headpose', 'stereo', 'ba']:
        if not isinstance(cfg['pools'][key], int) or cfg['pools'][key] < 1:
            raise RuntimeFailure('pool_size', 'Pool capacities must be positive integers')
    for kind in ['gpu', 'ram']:
        r = cfg['resources']
        if not 0 < r[kind + '_resume_gib'] <= r[kind + '_admit_gib'] < r[kind + '_hard_gib']:
            raise RuntimeFailure('admission', 'Require resume <= admit < hard')
    if path:
        base = Path(path).resolve().parent
        for item in cfg['assets'].values():
            if isinstance(item, dict) and item.get('path'):
                item['path'] = str((base / item['path']).resolve())
        for key, value in cfg['sources'].items():
            cfg['sources'][key] = str((base / value).resolve())
        if cfg['replay']['bundle']:
            cfg['replay']['bundle'] = str((base / cfg['replay']['bundle']).resolve())
    return cfg
