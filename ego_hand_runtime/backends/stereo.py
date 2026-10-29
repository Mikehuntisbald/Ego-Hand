"""Physical stereo and lazy mono fallback; surface depth is not a wrist anchor."""
import threading
import numpy as np
from ..geometry import rectified_pair, unproject, surface_intersections
from ..errors import RuntimeFailure
from ..assets import require_asset
from ..precision import geometry_precision,serialized_torch


def mono_anchor(depth, mask, vertices, joints, faces, camera, config):
    if mask is None:
        return dict(depth_valid=False, depth_m=None, depth_source='mono_missing_instance_mask')
    good = mask & np.isfinite(depth) & (depth>.08) & (depth<3.)
    yy, xx = np.where(good)
    if len(xx)<config['depth']['min_samples']:
        return dict(depth_valid=False, depth_m=None, depth_source='mono_coverage_rejected')
    ids = np.linspace(0, len(xx)-1, min(96, len(xx)), dtype=int)
    yy, xx = yy[ids], xx[ids]
    rays = unproject(np.stack([xx, yy], -1), camera)
    hits = surface_intersections(rays/rays[:, 2:3], vertices, faces)
    wrist = depth[yy, xx]-(hits-joints[0, 2])
    wrist = wrist[np.isfinite(hits)&(np.abs(hits-joints[0, 2])<.15)&(wrist>.08)&(wrist<3.)]
    median = float(np.median(wrist)) if len(wrist) else 0.
    mad = float(np.median(np.abs(wrist-median))) if len(wrist) else float('inf')
    valid = len(wrist)>=config['depth']['min_samples'] and mad<.03
    return dict(depth_valid=valid, depth_m=median if valid else None, depth_weight=.25,
                depth_source='UniDepth_surface_corrected_wrist', depth_MAD_m=mad, depth_samples=len(wrist))


class LightESS:
    def __init__(self, config):
        from .trt import Engine
        self.engine = Engine(config, 'lightess', 'ess_plugin')

    def __call__(self, left, right):
        import cv2
        values = {}
        for name, image in [('input_left', left), ('input_right', right)]:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            values[name] = np.repeat(gray[None, None], 3, axis=1).astype(np.float32)/127.5-1
        output = self.engine(values)
        if not all(k in output for k in ['output_left', 'output_conf']):
            raise RuntimeFailure('ESS_ABI', 'Engine does not expose disparity/confidence outputs')
        disparity,confidence=output['output_left'].squeeze(),output['output_conf'].squeeze()
        if disparity.shape!=left.shape[:2] or confidence.shape!=left.shape[:2] or not np.isfinite(disparity).all() or not np.isfinite(confidence).all():
            raise RuntimeFailure('ESS_numerics','LightESS backend returned an invalid shape or nonfinite values')
        return disparity,confidence


class UniDepth:
    @serialized_torch
    def __init__(self, config):
        # Import the model definition only in the actual Torch branch.
        import sys
        import torch
        from ..io import read_json
        source = config['sources'].get('unidepth')
        if not source:
            raise RuntimeFailure('unidepth_source', 'Pinned local UniDepth source is missing')
        sys.path.insert(0, source)
        from unidepth.models import UniDepthV2
        from safetensors.torch import load_file
        self.torch, self.device = torch, config['device']
        self.model = UniDepthV2(read_json(require_asset(config, 'unidepth_config'))).eval().to(self.device)
        self.model.load_state_dict(load_file(str(require_asset(config, 'unidepth'))), strict=True)
        self.model.resolution_level = 1

    def __call__(self, bgr, camera):
        import torch
        if camera.model!='fisheye624':
            raise RuntimeFailure('unidepth_camera', 'This public mono adapter only validates FISHEYE624')
        from unidepth.utils.camera import Fisheye624
        rgb = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1].transpose(2, 0, 1))).to(self.device)
        with geometry_precision(self.device),torch.inference_mode():
            params = torch.tensor(camera.parameters, device=self.device, dtype=torch.float32)[None]
            output = self.model.infer(rgb, Fisheye624(params=params))
        return output['depth'][0, 0].float().cpu().numpy()


def estimate(row, vertices, joints, faces, mask, episode, frame, decoder, engine, config, trace):
    import cv2
    primary = episode.camera_for_role('rgb')
    primary = episode.camera_at(primary.id, row['timestamp_ns'])
    options = [episode.camera_for_role(role) for role in ['tracking_left', 'tracking_right']]
    options = [episode.camera_at(c.id, row['timestamp_ns']) if c else None for c in options]
    values = []
    if mask is None:
        return dict(depth_valid=False, depth_m=None, depth_source='missing_instance_mask')
    center_ray = unproject((np.asarray(row['box_xyxy'][:2])+row['box_xyxy'][2:])/2, primary)
    image = decoder.read(frame['images'][primary.id])
    for target in options:
        source = frame['images'].get(target.id) if target else None
        if not source or not source['valid'] or abs(source['timestamp_ns']-frame['images'][primary.id]['timestamp_ns'])>config['sampling']['max_skew_ns']:
            continue
        w, h, f = [config['depth'][key] for key in ['width', 'height', 'focal_px']]
        base = rectified_pair(primary, target, w, h, f)
        native_from_rect = primary.T_device_from_camera[:3, :3].T@base['T_device_from_rect'][:3, :3]
        ray_rect = center_ray@native_from_rect
        if ray_rect[2]<=.05:
            continue
        spec = rectified_pair(primary, target, w, h, f, ray_rect[:2]/ray_rect[2])
        warped = [cv2.remap(img, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR)
                  for img, uv in zip([image, decoder.read(source)], spec['maps'])]
        with trace.span('stereo_model', camera=target.id):
            disparity, confidence = engine(*warped)
        yy, xx = np.mgrid[:h, :w]
        right_x = np.rint(xx-disparity).astype(int)
        good = spec['valid'][0] & (right_x>=0) & (right_x<w)
        good &= spec['valid'][1][yy, right_x.clip(0, w-1)]
        good &= np.isfinite(disparity)&(disparity>.5)&(confidence>=config['depth']['confidence_min'])
        z = f*spec['baseline_m']/np.maximum(disparity, 1e-6)
        principal = spec['principal']
        points = np.stack([(xx-principal[0])*z/f, (yy-principal[1])*z/f, z], -1)@native_from_rect.T
        uv = spec['maps'][0]
        px, py = np.rint(uv[..., 0]).astype(int), np.rint(uv[..., 1]).astype(int)
        good &= (px>=0)&(px<primary.width)&(py>=0)&(py<primary.height)
        good &= mask[py.clip(0, primary.height-1), px.clip(0, primary.width-1)]
        good &= (points[..., 2]>.08)&(points[..., 2]<3.)
        ids = np.flatnonzero(good)
        if len(ids)<config['depth']['min_samples']:
            continue
        ids = ids[np.linspace(0, len(ids)-1, min(96, len(ids)), dtype=int)]
        surface = points.reshape(-1, 3)[ids]
        hits = surface_intersections(surface/surface[:, 2:3], vertices, faces)
        offset = hits-joints[0, 2]
        wrist = surface[:, 2]-offset
        okay = np.isfinite(hits)&(np.abs(offset)<.15)&(wrist>.08)&(wrist<3.)
        values.extend(wrist[okay].tolist())
    if not values:
        return dict(depth_valid=False, depth_m=None, depth_source='stereo_rejected')
    median = float(np.median(values))
    mad = float(np.median(np.abs(np.asarray(values)-median)))
    good = len(values)>=config['depth']['min_samples'] and mad<.03
    return dict(depth_valid=good, depth_m=median if good else None, depth_MAD_m=mad,
                depth_samples=len(values), depth_weight=1., depth_source='LightESS_surface_corrected_wrist')
