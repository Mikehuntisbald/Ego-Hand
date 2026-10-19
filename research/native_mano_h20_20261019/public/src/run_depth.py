"""Public LightESS stereo -> ray/surface-corrected wrist-depth observations."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
import tensorrt as trt

sys.path.insert(0, '/mnt/why/public_headhand_h20_20261019/vendor/Ego-Hand/research')
from wilor_eval_common import unproject
from geometry import load_frames, camera, rectified_pair, remap_geometry, remap_image

import os
ROOT = Path(os.environ.get('PUBLIC_HEADHAND_ROOT','/mnt/why/public_headhand_h20_20261019'))


class LightESS:
    def __init__(self):
        source = ROOT/'assets/ess_legacy_public/dnn_stereo_disparity_v4.1.0_onnx'
        self.plugin = ctypes.CDLL(str(source/'plugins/x86_64/ess_plugins.so'), mode=ctypes.RTLD_GLOBAL)
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine((ROOT/'assets/lightess_h20_fp16.engine').read_bytes())
        assert self.engine is not None
        self.context = self.engine.create_execution_context()
        self.buffers = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
            self.buffers[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)), device='cuda:0', dtype=dtype)
            self.context.set_tensor_address(name, self.buffers[name].data_ptr())
        self.stream = torch.cuda.Stream()

    def __call__(self, left, right):
        with torch.cuda.stream(self.stream):
            for name, image in [('input_left', left), ('input_right', right)]:
                # Both Aria streams are converted to luminance for photometric
                # consistency, then replicated to the model's RGB channel contract.
                if image.ndim == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rgb = np.repeat(image[..., None], 3, axis=-1)
                value = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1)))[None]
                self.buffers[name].copy_(value.to('cuda:0').float()/127.5-1.)
            assert self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return [self.buffers[name].float().cpu().numpy()[0].copy() for name in ['output_left', 'output_conf']]


def surface_intersections(directions, vertices, faces):
    """Two-sided Moller-Trumbore; directions have Z=1, so the hit t is Z."""
    triangles = vertices[faces]
    v0 = triangles[:, 0]
    edge1, edge2 = triangles[:, 1]-v0, triangles[:, 2]-v0
    p = np.cross(directions[:, None], edge2[None])
    determinant = (p*edge1[None]).sum(-1)
    inverse = np.divide(1., determinant, out=np.zeros_like(determinant), where=np.abs(determinant) > 1e-9)
    u = (p*(-v0)[None]).sum(-1)*inverse
    q = np.cross(-v0, edge1)
    v = (directions[:, None]*q[None]).sum(-1)*inverse
    distance = (edge2*q).sum(-1)[None]*inverse
    valid = (np.abs(determinant) > 1e-9) & (u >= 0) & (v >= 0) & (u+v <= 1) & (distance > .01)
    return np.where(valid, distance, np.inf).min(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--coarse', type=Path, required=True)
    parser.add_argument('--rf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False, parents=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    frames = load_frames(args.input)
    coarse = json.loads((args.coarse/'observations.json').read_text())['frames']
    geometry = np.load(args.coarse/'surface.npz')
    rf = json.loads((args.rf/'predictions.json').read_text())['frames']
    model = LightESS()
    results = []
    start = time.time()
    for frame_index in range(0, len(frames), 2):
        frame = frames[frame_index]
        ids = [i for i, row in enumerate(coarse) if row['frame_index'] == frame_index]
        if not ids:
            continue
        with np.load(rf[frame_index]['output']) as values:
            masks = values['masks']
        own_masks = masks & ~(masks.sum(0) > 1)[None]
        rgb = cv2.imread(frame['images']['214-1'])
        native = camera(frame['cameras']['214-1']['calibration'])
        uv_centers = np.array([(np.array(coarse[i]['box_xyxy'][:2])+coarse[i]['box_xyxy'][2:])/2 for i in ids])
        central_rays = np.stack([unproject(native, uv) for uv in uv_centers])
        samples = {i: [] for i in ids}
        for target in ['1201-1', '1201-2']:
            spec = rectified_pair(frame['cameras'], '214-1', target)
            native_from_rect = spec['first_device_transform'][:3, :3].T @ spec['device_from_rectified']
            ray_rect = central_rays @ native_from_rect
            visible = ray_rect[:, 2] > .05
            if not visible.any():
                continue
            center = np.median(ray_rect[visible, :2]/ray_rect[visible, 2:3], axis=0)
            principal = np.array([239.5, 143.5])-spec['focal_px']*center
            maps, valid = [], []
            for key in ['214-1', target]:
                from geometry import transform
                pose = transform(frame['cameras'][key]['calibration']['T_device_from_camera'])
                rotation = pose[:3, :3].T @ spec['device_from_rectified']
                uv, good = remap_geometry(frame['cameras'][key]['calibration'], 480, 288,
                                           spec['focal_px'], principal, rotation)
                maps.append(uv)
                valid.append(good)
            side = cv2.imread(frame['images'][target])
            disparity, confidence = model(remap_image(rgb, maps[0]), remap_image(side, maps[1]))
            yy, xx = np.mgrid[:288, :480]
            right_x = np.rint(xx-disparity).astype(np.int32)
            supported = (right_x >= 0) & (right_x < 480)
            supported &= valid[1][yy, np.clip(right_x, 0, 479)]
            good = valid[0] & supported & (confidence >= .2) & (disparity > .5) & np.isfinite(disparity)
            z = spec['focal_px']*spec['baseline_m']/np.maximum(disparity, 1e-6)
            points = np.stack([(xx-principal[0])*z/spec['focal_px'],
                               (yy-principal[1])*z/spec['focal_px'], z], -1) @ native_from_rect.T
            native_uv = maps[0]
            px = np.rint(native_uv[..., 0]).astype(np.int32)
            py = np.rint(native_uv[..., 1]).astype(np.int32)
            supported_native = (px >= 0) & (px < 1408) & (py >= 0) & (py < 1408)
            good &= supported_native & (points[..., 2] > .08) & (points[..., 2] < 3.)
            for i in ids:
                mask = own_masks[coarse[i]['detection']]
                selected = good & mask[np.clip(py, 0, 1407), np.clip(px, 0, 1407)]
                locations = np.flatnonzero(selected)
                if len(locations) < 6:
                    continue
                locations = locations[np.linspace(0, len(locations)-1, min(96, len(locations)), dtype=int)]
                surface = points.reshape(-1, 3)[locations].astype(np.float32)
                directions = surface/surface[:, 2:3]
                hits = surface_intersections(directions, geometry['vertices'][i], geometry['faces'])
                offset = hits-geometry['joints'][i, 0, 2]
                accepted = np.isfinite(hits) & (np.abs(offset) < .15)
                wrist_z = surface[:, 2]-offset
                accepted &= (wrist_z > .08) & (wrist_z < 3.)
                samples[i].extend(wrist_z[accepted].tolist())
            if frame_index == 0:
                np.savez_compressed(args.output/f'example_{target}.npz', disparity=disparity, confidence=confidence,
                                    valid=good, left=remap_image(rgb, maps[0]), right=remap_image(side, maps[1]))
        for i in ids:
            values = np.array(samples[i])
            median = float(np.median(values)) if len(values) else None
            mad = float(np.median(np.abs(values-median))) if len(values) else None
            valid_anchor = len(values) >= 6 and mad < .03
            results.append(dict(observation=i, frame_index=frame_index, track_id=coarse[i]['track_id'],
                                timestamp_ns=frame['timestamp_ns'], wrist_depth_m=median if valid_anchor else None,
                                valid=bool(valid_anchor), samples=len(values), MAD_m=mad,
                                source='public-LightESS-stereo-surface-corrected',
                                original_wrist_z_m=coarse[i]['state_camera51'][2]))
        if frame_index % 20 == 0:
            print(json.dumps(dict(frame=frame_index, anchors=sum(x['valid'] for x in results),
                                  observations=len(results), elapsed_s=time.time()-start)), flush=True)
    report = dict(complete=True, observations=results, GT_used=False, confidence_threshold=.2,
                  stereo_stride=2, public_input=[480, 288], surface='MANO triangle/ray intersection; wrist-center offset removed',
                  rectification='Physical extrinsics; shared ROI principal for left/right; calibrated focal and baseline',
                  seconds=time.time()-start, valid=sum(x['valid'] for x in results), total=len(results))
    path = args.output/'depth.json'
    path.write_text(json.dumps(report, indent=2))
    (args.output/'seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), GT_used=False), indent=2))
    print(json.dumps(dict(complete=True, valid=report['valid'], total=len(results), seconds=report['seconds'])), flush=True)


if __name__ == '__main__':
    main()
