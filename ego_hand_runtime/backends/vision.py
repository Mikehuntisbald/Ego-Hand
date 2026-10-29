import importlib
from pathlib import Path
import sys
import numpy as np
from ..assets import require_asset
from ..errors import RuntimeFailure
from ..geometry import perspective_crop, project, pixel_mask, matrix_rotvec
from ..precision import serialized_torch


class Detector:
    @serialized_torch
    def __init__(self, config):
        from rfdetr import RFDETR
        from ultralytics import YOLO
        self.device = config['device']
        self.model = RFDETR.from_checkpoint(str(require_asset(config, 'rf')), device=self.device, trust_checkpoint=True)
        self.side = YOLO(str(require_asset(config, 'side_helper')))

    @serialized_torch
    def predict(self, bgr):
        import torch
        from PIL import Image
        with torch.inference_mode():
            detections = self.model.predict(Image.fromarray(bgr[..., ::-1].copy()), threshold=.2, include_source_image=False)
        if isinstance(detections, list):
            detections = detections[0]
        side = self.side.predict(bgr, imgsz=960, conf=.001, max_det=100,
                                 device=self.device.replace('cuda:', ''), verbose=False)[0]
        boxes = np.asarray(detections.xyxy)
        scores = np.asarray(detections.confidence)
        masks = getattr(detections, 'mask', None)
        if masks is not None:
            masks=np.asarray(masks,bool)
            if masks.shape!=(len(boxes),*bgr.shape[:2]):
                raise RuntimeFailure('RF_mask_ABI','RF instance masks must be on the actual source image grid')
            # Ambiguous overlap pixels are excluded from every physical-depth
            # instance; they must not borrow surface depth from the other hand.
            masks=masks & ~(masks.sum(0)>1)[None]
        hboxes = side.boxes.xyxy.cpu().numpy()
        hscores = side.boxes.conf.cpu().numpy()
        hclasses = side.boxes.cls.cpu().numpy()
        output = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            overlap = iou(boxes[i:i+1], hboxes).reshape(-1)
            chosen = int(overlap.argmax()) if len(overlap) else None
            matched = chosen is not None and overlap[chosen]>=.05
            output.append(dict(box_xyxy=box.tolist(), confidence=float(score),
                               right=bool(hclasses[chosen]) if matched else True,
                               side_confidence=float(hscores[chosen]) if matched else 0.,
                               side_uncertain=not matched, detection_index=i))
        return output, masks


def iou(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    low = np.maximum(a[:, None, :2], b[None, :, :2])
    high = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(high-low, 0).prod(-1)
    return intersection/np.maximum(np.maximum(a[:, 2:]-a[:, :2], 0).prod(-1)[:, None]+
                                  np.maximum(b[:, 2:]-b[:, :2], 0).prod(-1)[None]-intersection, 1e-8)


def associate_and_vote(rows, min_side_confidence=.5):
    from scipy.optimize import linear_sum_assignment
    active, next_id = {}, 0
    for stamp in sorted({r['timestamp_ns'] for r in rows}):
        current = [r for r in rows if r['timestamp_ns']==stamp]
        old = [(k, v) for k, v in active.items() if 0<stamp-v['timestamp_ns']<=510000000]
        assignments = {}
        if current and old:
            overlaps = iou([r['box_xyxy'] for r in current], [v['box_xyxy'] for _, v in old])
            a, b = linear_sum_assignment(1-overlaps)
            for x, y in zip(a, b):
                if overlaps[x, y]>.1:
                    assignments[x] = old[y][0]
        for index, row in enumerate(current):
            track = assignments.get(index)
            if track is None:
                track, next_id = next_id, next_id+1
            row['track_id'] = track
            row['owner'] = 'unknown'
            active[track] = row
    for track in range(next_id):
        values = [r for r in rows if r['track_id']==track]
        votes = [sum(r['side_confidence'] for r in values if r['right']==side) for side in [False, True]]
        margin = abs(votes[1]-votes[0])/max(sum(votes), 1e-8)
        side = votes[1]>=votes[0]
        support = sum(r['right']==side and r['side_confidence']>=min_side_confidence for r in values)
        if margin>=.6 and support>=3:
            for r in values:
                r['original_right'] = r['right']
                r['right'], r['side_uncertain'] = side, False
                r['side_source'] = 'whole-track predicted confidence before actual WiLoR forward'
    return rows


class WiLoR:
    @serialized_torch
    def __init__(self, config):
        import torch
        self.config, self.device = config, torch.device(config['device'])
        source = config['sources'].get('wilor')
        if not source or not Path(source).is_dir():
            raise RuntimeFailure('source', 'Pinned local WiLoR source is missing')
        sys.path.insert(0, source)
        from wilor.configs import get_config
        from wilor.models import WiLoR as Network
        cfg = get_config(str(require_asset(config, 'wilor_config')))
        cfg.defrost()
        cfg.MODEL.BBOX_SHAPE = [192, 256]
        cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS', None)
        cfg.MANO.MODEL_PATH = str(require_asset(config, 'mano_right'))
        cfg.MANO.DATA_DIR = str(require_asset(config, 'mano_mean').parent)
        cfg.MANO.MEAN_PARAMS = str(require_asset(config, 'mano_mean'))
        cfg.LOSS_WEIGHTS.ADVERSARIAL = 0
        cfg.freeze()
        from ..native import legacy_mano_pickle
        with legacy_mano_pickle():
            self.model = Network(cfg, init_renderer=False)
        ck = torch.load(require_asset(config, 'wilor'), map_location='cpu', weights_only=False)
        mismatch = self.model.load_state_dict(ck.get('state_dict', ck), strict=False)
        if mismatch.missing_keys or any(not k.startswith('discriminator.') for k in mismatch.unexpected_keys):
            raise RuntimeFailure('wilor_weights', 'WiLoR source/checkpoint ABI mismatch')
        self.model.to(self.device).eval()

    @serialized_torch
    def predict(self, image, row, camera, codec):
        import torch
        warped, crop_R, focal = perspective_crop(image, row['box_xyxy'], camera)
        warped = np.rot90(warped, 1)
        if not row['right']:
            warped = warped[:, ::-1]
        rgb = warped[..., ::-1].astype(np.float32)/255
        tensor = torch.from_numpy(np.ascontiguousarray(((rgb-np.array([.485, .456, .406]))/
                     np.array([.229, .224, .225])).transpose(2, 0, 1))).to(self.device, dtype=torch.float32)[None]
        with torch.inference_mode(), torch.autocast(device_type=self.device.type, dtype=torch.float16,
                                                   enabled=self.device.type=='cuda'):
            prediction = self.model({'img': tensor})
        params = prediction['pred_mano_params']
        root = params['global_orient'].float().reshape(3, 3).cpu().numpy()
        local = params['hand_pose'].float().reshape(15, 3, 3).cpu().numpy()
        beta = params['betas'].float().reshape(10).cpu().numpy()
        weak = prediction['pred_cam'].float().reshape(3).cpu().numpy()
        raw = prediction['pred_keypoints_3d'].float().reshape(21, 3).cpu().numpy()
        sign = 1 if row['right'] else -1
        raw[:, 0] *= sign
        weak[1] *= sign
        shift = np.array([weak[1], weak[2], 2*focal/(256*max(float(weak[0]), 1e-6))], np.float32)
        Q = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], np.float32)
        geometry = crop_R@Q.T
        xyz = (raw+shift)@geometry.T
        S = np.diag([sign, 1, 1])
        # BF16/FP16 rotations are projected once to proper SO(3) via the same
        # rotvec conversion as the frozen front end; all geometry afterward is FP32.
        state = np.r_[xyz[0], matrix_rotvec(geometry@S@root@S), matrix_rotvec(local).reshape(-1)].astype(np.float32)
        joints, vertices = codec(state[None], beta[None], np.array([row['right']]), vertices=True)
        uv = project(joints[0], camera)
        output = dict(row, state_camera51=state.tolist(), betas=beta.tolist(), uv=uv.tolist(),
                      point_valid=pixel_mask(joints[0], uv, camera).tolist(),
                      observation_kind='wilor_self_projection', valid=not row.get('side_uncertain', False),
                      box_scale_px=float(max(np.asarray(row['box_xyxy'])[2:]-np.asarray(row['box_xyxy'])[:2])),
                      FK_rounding_difference_m=float(np.linalg.norm(joints[0]-xyz, axis=-1).max()))
        return output, vertices[0], joints[0]
