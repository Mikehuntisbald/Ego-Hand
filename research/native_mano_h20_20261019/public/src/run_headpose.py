"""Real public cuVSLAM stereo inference on sanitized HOT3D camera inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import cuvslam
from geometry import load_frames, transform, remap_geometry, remap_image, rectified_pair


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=150)
    parser.add_argument('--rectified', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    frames = load_frames(args.input)[:args.frames]
    calibration = frames[0]['cameras']
    keys = ['1201-1', '1201-2']
    physical = [transform(calibration[key]['calibration']['T_device_from_camera']) for key in keys]
    rig_from_cameras = [np.linalg.inv(physical[0]) @ item for item in physical]
    maps = [remap_geometry(calibration[key]['calibration'], 640, 480, 240.)[0] for key in keys]
    focal=240.
    virtual_from_left=np.eye(4)
    if args.rectified:
        spec=rectified_pair(calibration,*keys,width=640,height=480,focal=180.)
        maps=spec['maps'];focal=180.
        # Rig becomes the left rectified camera; right rectified axes are parallel.
        rig_from_cameras=[np.eye(4),np.eye(4)]
        rig_from_cameras[1][0,3]=spec['baseline_m']
        virtual_from_left[:3,:3]=spec['device_from_rectified'].T@physical[0][:3,:3]
    cameras = []
    for matrix in rig_from_cameras:
        cam = cuvslam.Camera()
        cam.size = [640, 480]
        cam.focal = [focal, focal]
        cam.principal = [319.5, 239.5]
        cam.rig_from_camera = cuvslam.Pose(rotation=Rotation.from_matrix(matrix[:3, :3]).as_quat(),
                                           translation=matrix[:3, 3])
        cameras.append(cam)
    config = cuvslam.Tracker.OdometryConfig(async_sba=False, rectified_stereo_camera=args.rectified)
    tracker = cuvslam.Tracker(cuvslam.Rig(cameras), config)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    observations = []
    start = time.time()
    for index, frame in enumerate(frames):
        times = [int(frame['image_timestamps_ns'][key]) for key in keys]
        assert abs(times[1]-times[0]) <= 5_000_000, 'Stereo skew exceeds 5ms'
        images = [np.ascontiguousarray(clahe.apply(remap_image(cv2.imread(frame['images'][key], 0), uv)))
                  for key, uv in zip(keys, maps)]
        timestamp = (times[0]+times[1])//2
        estimate, _ = tracker.track(timestamp, images)
        # Copy SDK memory immediately. No reference pose or hand labels enter.
        valid = estimate.world_from_rig is not None
        item = dict(frame=frame['id'], timestamp_ns=timestamp, valid=valid)
        if valid:
            pose = estimate.world_from_rig.pose
            q = np.array(pose.rotation, dtype=float).copy()
            t = np.array(pose.translation, dtype=float).copy()
            matrix = np.eye(4)
            matrix[:3, :3] = Rotation.from_quat(q).as_matrix()
            matrix[:3, 3] = t
            item['T_world_from_left'] = matrix.tolist()
        observations.append(item)
        if index % 25 == 0:
            print(json.dumps(dict(frame=index, valid=valid, elapsed_s=time.time()-start)), flush=True)
    valid = [item for item in observations if item['valid']]
    assert len(valid) > 2, 'Too few valid stereo poses'
    times = np.array([item['timestamp_ns'] for item in valid], dtype=np.int64)
    matrices = np.array([item['T_world_from_left'] for item in valid])
    # Use relative seconds to avoid precision loss from large nanosecond epochs.
    seconds = (times-times[0])/1e9
    rotations = Slerp(seconds, Rotation.from_matrix(matrices[:, :3, :3]))
    left_from_rgb = virtual_from_left @ np.linalg.inv(physical[0]) @ transform(calibration['214-1']['calibration']['T_device_from_camera'])
    output = []
    for frame in frames:
        ns = frame['timestamp_ns']
        insertion = int(np.searchsorted(times, ns))
        nearest = int(np.abs(times-ns).argmin())
        good = abs(int(times[nearest])-ns) <= 40_000_000
        if 0 < insertion < len(times):
            good = good and times[insertion]-times[insertion-1] <= 100_000_000
        outside = ns < times[0] or ns > times[-1]
        if outside:
            good = good and abs(int(times[nearest])-ns) <= 2_000_000
        item = dict(frame=frame['id'], timestamp_ns=ns, valid=bool(good), segment=0,
                    reference_pose_used=False, temporal_boundary_hold=bool(outside))
        if good:
            query = np.clip((ns-times[0])/1e9, seconds[0], seconds[-1])
            pose = np.eye(4)
            pose[:3, :3] = rotations(query).as_matrix()
            pose[:3, 3] = [np.interp(query, seconds, matrices[:, dimension, 3]) for dimension in range(3)]
            item['T_world_from_rgb'] = (pose @ left_from_rgb).tolist()
        output.append(item)
    result = dict(backend='public cuVSLAM17 pure stereo', dataset='HOT3D Aria clip2102',
                  source_commit='57f42cc92d93eef47577d726789852a350b2a369',
                  frames=output, native_stereo_poses=observations, seconds=time.time()-start,
                  world_gauge='First stereo rig frame; no scale/alignment reference input',
                  CLAHE='CPU OpenCV; not the internal CUDA preprocessing implementation',
                  count=len(output), valid=sum(item['valid'] for item in output), GT_used=False,
                  near_LK_custom_SDK=False)
    result['common_view_rectified']=args.rectified
    path = args.output/'headpose.json'
    path.write_text(json.dumps(result, indent=2))
    (args.output/'seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                                       prediction_frozen=True, GT_used=False), indent=2))
    print(json.dumps(dict(complete=True, valid=result['valid'], frames=len(output), seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
