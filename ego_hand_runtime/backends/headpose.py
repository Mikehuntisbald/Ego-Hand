"""Actual cuVSLAM17 stereo VO, with loaded-library evidence and motion validity."""
from pathlib import Path
import os
import numpy as np
from ..errors import RuntimeFailure
from ..io import sha256
from ..geometry import rectified_pair, quaternion_matrix, matrix_rotvec,virtual_camera_map
from collections import OrderedDict
from ..io import identity


def loaded_sdk(module):
    paths = set()
    maps = Path('/proc/self/maps')
    if maps.exists():
        for line in maps.read_text().splitlines():
            candidate = line.split()[-1]
            if 'cuvslam' in candidate.lower() and Path(candidate).is_file():
                paths.add(candidate)
    else:
        root = Path(module.__file__).parent
        paths.update(str(p) for p in root.rglob('*') if p.is_file() and p.suffix in {'.so', '.pyd', '.dll'})
    return [dict(path=p, sha256=sha256(p)) for p in sorted(paths)]


class StereoVO:
    def __init__(self, config):
        import cuvslam
        self.sdk, self.config = cuvslam, config
        self.loaded_libraries = loaded_sdk(cuvslam)
        expected = config['headpose']['expected_sdk_sha256']
        if not expected or expected not in {x['sha256'] for x in self.loaded_libraries}:
            raise RuntimeFailure('sdk_sha', 'Loaded cuVSLAM library has not passed the configured SHA pin',
                                 loaded=self.loaded_libraries)

    def run(self, episode, decoder, trace, guard=None):
        import cv2
        import ctypes
        runtime=next((p['path'] for p in self.loaded_libraries if 'libcudart' in p['path']),None)
        if runtime is None:
            raise RuntimeFailure('sdk_device','Cannot establish the actually loaded CUDA runtime for SDK device selection')
        cuda=ctypes.CDLL(runtime)
        cuda.cudaSetDevice.argtypes=[ctypes.c_int];cuda.cudaSetDevice.restype=ctypes.c_int
        if cuda.cudaSetDevice(int(self.config['device'].split(':')[-1]))!=0:
            raise RuntimeFailure('sdk_device','Requested logical CUDA device is unavailable')
        sdk = self.sdk
        left, right = episode.camera_for_role('tracking_left'), episode.camera_for_role('tracking_right')
        if left is None or right is None:
            raise RuntimeFailure('stereo_camera', 'Real left/right tracking cameras are required')
        spec = rectified_pair(left, right, 640, 480, 180.)
        cameras = []
        from scipy.spatial.transform import Rotation
        for x in [0., spec['baseline_m']]:
            cam = sdk.Camera()
            cam.size, cam.focal, cam.principal = [640, 480], [180., 180.], [319.5, 239.5]
            cam.rig_from_camera = sdk.Pose(rotation=[0., 0., 0., 1.], translation=[x, 0., 0.])
            cameras.append(cam)
        # A fresh tracker state per episode is mandatory even with a persistent SDK slot.
        tracker = sdk.Tracker(sdk.Rig(cameras), sdk.Tracker.OdometryConfig(use_gpu=True,async_sba=False, rectified_stereo_camera=True))
        clahe = cv2.createCLAHE(clipLimit=2., tileGridSize=(8, 8))
        observations, previous, segment = [], None, 0
        drift_m, drift_rad = 0., 0.
        map_cache=OrderedDict();map_updates=0
        for frame in episode.frames:
            if guard is not None:guard()
            # Public SDK rig is fixed for an episode. Admit only explicitly
            # bounded online-calibration drift, and report the approximation.
            maps=[]
            for camera in [left, right]:
                actual = frame.get('cameras', episode.cameras)[camera.id]
                delta = np.linalg.inv(camera.T_device_from_camera)@actual.T_device_from_camera
                drift_m = max(drift_m, float(np.linalg.norm(delta[:3,3])))
                drift_rad = max(drift_rad, float(np.linalg.norm(matrix_rotvec(delta[:3,:3]))))
                key=identity(actual.to_dict())
                if key not in map_cache:
                    map_cache[key]=virtual_camera_map(actual,spec['T_device_from_rect'][:3,:3],
                                                      640,480,180.,[319.5,239.5])[0]
                    map_updates+=1
                    while len(map_cache)>4:map_cache.popitem(last=False)
                map_cache.move_to_end(key);maps.append(map_cache[key])
            if drift_m>self.config['headpose']['max_rig_drift_m'] or drift_rad>self.config['headpose']['max_rig_drift_rad']:
                raise RuntimeFailure('head_calibration', 'Tracking rig drift exceeds declared fixed-rig tolerance; split explicitly')
            a, b = frame['images'].get(left.id), frame['images'].get(right.id)
            stamp = (a['timestamp_ns']+b['timestamp_ns'])//2 if a and b else frame['timestamp_ns']
            item = dict(timestamp_ns=stamp, source_timestamp_ns=frame['timestamp_ns'],
                        source_index=frame['source_index'], valid=False, segment=segment)
            if a and b and a['valid'] and b['valid'] and abs(a['timestamp_ns']-b['timestamp_ns'])<=self.config['sampling']['max_skew_ns']:
                with trace.span('headpose_decode_preprocess'):
                    images = []
                    for source, uv in zip([a, b], maps):
                        gray = cv2.cvtColor(decoder.read(source), cv2.COLOR_BGR2GRAY)
                        images.append(np.ascontiguousarray(clahe.apply(cv2.remap(gray, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR))))
                measured_stamp = (a['timestamp_ns']+b['timestamp_ns'])//2
                with trace.span('headpose_model'):
                    estimate, _ = tracker.track(measured_stamp, images)
                if estimate.world_from_rig is not None:
                    pose = estimate.world_from_rig.pose
                    T = np.eye(4)
                    T[:3, :3] = quaternion_matrix(np.array(pose.rotation).copy())
                    T[:3, 3] = np.array(pose.translation).copy()
                    world_from_device = T@np.linalg.inv(spec['T_device_from_rect'])
                    good = np.isfinite(world_from_device).all()
                    if previous is not None:
                        dt = (measured_stamp-previous[0])/1e9
                        good = good and dt>0
                        if good:
                            speed = np.linalg.norm(world_from_device[:3, 3]-previous[1][:3, 3])/dt
                            angular = np.linalg.norm(matrix_rotvec(previous[1][:3, :3].T@world_from_device[:3, :3]))/dt
                            good = speed<=self.config['headpose']['max_speed_m_s'] and angular<=self.config['headpose']['max_angular_speed_rad_s']
                    if good:
                        item.update(timestamp_ns=measured_stamp, valid=True, world_from_device=world_from_device.tolist())
                        previous = measured_stamp, world_from_device
                    else:
                        segment += 1
                        item.update(segment=segment, failure='motion_guard')
                        previous = None
            if not item['valid']:
                previous = None
                segment += 1
            if observations and item['timestamp_ns'] <= observations[-1]['timestamp_ns']:
                raise RuntimeFailure('head_clock', 'Stereo source clock is not increasing; split/reset explicitly')
            observations.append(item)
        if sum(x['valid'] for x in observations)<2:
            raise RuntimeFailure('headpose_failed', 'Too few valid measured stereo VO poses')
        # Return measured clock. Delivery aligns onto the source clock with bounded
        # interpolation; neither GT poses nor unrestricted edge holds are used.
        return observations, dict(backend='public_cuvslam17_pure_stereo', libraries=self.loaded_libraries,
                                  custom_near=False, IMU_fusion=False, GT_used=False,
                                  fixed_rig_calibration_drift_m=drift_m, fixed_rig_calibration_drift_rad=drift_rad,
                                  physical_calibration='per-frame lens/orientation maps onto fixed virtual rig; origin drift bounded',
                                  physical_map_updates=map_updates,physical_map_cache_capacity=4)
