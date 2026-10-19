"""Public WiLoR -> native-camera MANO21 state, preserving actual crop geometry."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

REFERENCE = Path('/mnt/why/public_headhand_h20_20261019/vendor/Ego-Hand/research')
sys.path.insert(0, str(REFERENCE))
import wilor_eval_common as common
from geometry import camera, load_frames, IDENTITY
from mano_model import WristAnchoredMANO


def iou(a, b):
    a, b = np.asarray(a).reshape(-1, 4), np.asarray(b).reshape(-1, 4)
    lower = np.maximum(a[:, None, :2], b[None, :, :2])
    upper = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(upper-lower, 0).prod(-1)
    area = np.maximum(a[:, 2:]-a[:, :2], 0).prod(-1)
    other = np.maximum(b[:, 2:]-b[:, :2], 0).prod(-1)
    return intersection/np.maximum(area[:, None]+other[None]-intersection, 1e-8)


def assign_tracks_and_sides(observations):
    active, next_id = {}, 0
    for fi in sorted({r['frame_index'] for r in observations}):
        ids=[i for i,r in enumerate(observations) if r['frame_index']==fi]
        stamp=observations[ids[0]]['timestamp_ns']
        previous=[(key,value) for key,value in active.items() if (stamp-value['timestamp_ns'])/1e9<=.51]
        assignments={}
        if ids and previous:
            overlap=iou([observations[i]['box_xyxy'] for i in ids],[value['box_xyxy'] for key,value in previous])
            a,b=linear_sum_assignment(1-overlap)
            for x,y in zip(a,b):
                if overlap[x,y]>.1:assignments[ids[x]]=previous[y][0]
        for i in ids:
            tid=assignments.get(i)
            if tid is None:tid,next_id=next_id,next_id+1
            observations[i]['track_id']=tid;active[tid]=observations[i]
    decisions=[]
    for tid in sorted({r['track_id'] for r in observations}):
        ids=[i for i,r in enumerate(observations) if r['track_id']==tid]
        votes=np.array([sum(observations[i]['side_confidence'] for i in ids if observations[i]['right']==side) for side in [0,1]])
        side=int(votes.argmax());margin=float(abs(votes[1]-votes[0])/max(votes.sum(),1e-8))
        support=sum(observations[i]['right']==side and observations[i]['side_confidence']>=.5 for i in ids)
        accepted=margin>=.6 and support>=3
        decisions.append(dict(track_id=tid,votes=votes.tolist(),margin=margin,strong_support=support,assigned_right=side if accepted else None))
        if accepted:
            for i in ids:
                row=observations[i];row['original_right']=row['right'];row['right']=side
                row['side_source']='prediction-only whole-track confidence vote before WiLoR inference'
    return next_id, decisions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--rf', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    frames = load_frames(args.input)
    rf = json.loads((args.rf/'predictions.json').read_text())
    assert rf['GT_free'] and len(rf['frames']) == len(frames)
    from ultralytics import YOLO
    helper_path = '/mnt/why/HOT3D/experiments/detector_compare_wilor_20261003/wilor_detector.pt'
    helper = YOLO(helper_path)
    sides = {}
    for begin in range(0, len(frames), 16):
        batch = frames[begin:begin+16]
        paths = [row['images']['214-1'] for row in batch]
        output = helper.predict(paths, imgsz=960, conf=.001, max_det=100, device=0, verbose=False)
        for frame, value in zip(batch, output):
            sides[frame['id']] = dict(boxes=value.boxes.xyxy.cpu().numpy(),
                                     scores=value.boxes.conf.cpu().numpy(), classes=value.boxes.cls.cpu().numpy())
    del helper
    torch.cuda.empty_cache()
    observations, crops, matrices, focals, rights = [], [], [], [], []
    for index, (frame, record) in enumerate(zip(frames, rf['frames'])):
        image = cv2.imread(frame['images']['214-1'])
        native = dict(calibration=frame['cameras']['214-1']['calibration'], T_world_from_camera=IDENTITY)
        with np.load(record['output']) as values:
            boxes, scores = values['boxes'], values['scores']
        helper = sides[frame['id']]
        overlap = iou(boxes, helper['boxes'])
        for detection, (box, score) in enumerate(zip(boxes, scores)):
            side, confidence = 1, 0.
            if len(helper['boxes']):
                chosen = int(overlap[detection].argmax())
                if overlap[detection, chosen] >= .05:
                    side, confidence = int(helper['classes'][chosen]), float(helper['scores'][chosen])
            crop, rotation, focal, _ = common.crop(image, box, native, padding=1.3)
            crops.append(crop)
            matrices.append(rotation)
            focals.append(focal)
            rights.append(side)
            observations.append(dict(frame_index=index, detection=detection, frame=frame['id'],
                                     timestamp_ns=frame['timestamp_ns'], image=frame['images']['214-1'],
                                     box_xyxy=box.tolist(), confidence=float(score), right=side,
                                     side_confidence=confidence, side_review_required=confidence < .5,
                                     owner='unknown'))
    assert observations, 'No detected hands'
    next_id, decisions = assign_tracks_and_sides(observations)
    rights = [row['right'] for row in observations]
    model, _ = common.load_model('cuda:0')
    decoder = WristAnchoredMANO(model.mano)
    captured = {}
    hook = model.backbone.register_forward_hook(lambda module, inp, out: captured.update(features=out[-1]))
    states, shapes, joints, features, crop_rotations, crop_focals = [], [], [], [], [], []
    maximum_parity = 0.
    started = time.time()
    for begin in range(0, len(crops), 16):
        end = min(begin+16, len(crops))
        flags = np.array(rights[begin:end])
        tensor = common.input_tensor(crops[begin:end], flags, 1).to('cuda:0')
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            output = model({'img': tensor})
        native_joints = output['pred_keypoints_3d'].float().cpu().numpy()
        parameters = output['pred_mano_params']
        global_rotation = parameters['global_orient'].float().reshape(-1, 3, 3).cpu().numpy()
        local_rotation = parameters['hand_pose'].float().reshape(-1, 15, 3, 3).cpu().numpy()
        betas = parameters['betas'].float().reshape(-1, 10).cpu().numpy()
        camera_parameters = output['pred_cam'].float().cpu().numpy()
        sign = flags*2-1
        native_joints[..., 0] *= sign[:, None]
        camera_parameters[:, 1] *= sign
        translation = np.stack([camera_parameters[:, 1], camera_parameters[:, 2],
                                2*np.asarray(focals[begin:end])/(256*np.maximum(camera_parameters[:, 0], 1e-6))], -1)
        quarter_turn = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], np.float32)
        geometry_rotation = np.asarray(matrices[begin:end]) @ quarter_turn.T
        xyz = np.einsum('bij,bkj->bki', geometry_rotation, native_joints+translation[:, None])
        reflection = np.tile(np.eye(3), (end-begin, 1, 1))
        reflection[:, 0, 0] = sign
        physical_rotation = geometry_rotation @ reflection @ global_rotation @ reflection
        state = np.concatenate([xyz[:, 0], Rotation.from_matrix(physical_rotation).as_rotvec(),
                                Rotation.from_matrix(local_rotation.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 45)], -1).astype(np.float32)
        with torch.inference_mode():
            recovered = decoder(torch.from_numpy(state).cuda(), torch.from_numpy(betas).cuda(),
                                torch.from_numpy(flags).cuda()).cpu().numpy()
        # BF16 model's stored rotations can be slightly non-orthogonal; rotvec
        # conversion projects them to SO(3). Record rather than hide this difference.
        parity = float(np.max(np.linalg.norm(recovered-xyz, axis=-1)))
        maximum_parity = max(maximum_parity, parity)
        states.append(state)
        shapes.append(betas)
        joints.append(recovered)
        features.append(captured['features'].float().flatten(2).transpose(1, 2).half().cpu().numpy())
        crop_rotations.append(geometry_rotation)
        crop_focals.extend(focals[begin:end])
        print(json.dumps(dict(observations=end, total=len(crops), FK_difference_mm=parity*1000)), flush=True)
    hook.remove()
    states, shapes, joints, features = [np.concatenate(values) for values in [states, shapes, joints, features]]
    for i,row in enumerate(observations):
        native = camera(frames[row['frame_index']]['cameras']['214-1']['calibration'])
        row['xy'] = native.eye_to_window(joints[i]).tolist()
        row['state_camera51'] = states[i].tolist()
        row['betas10'] = shapes[i].tolist()
    np.savez_compressed(args.output/'observations.npz', state=states, betas=shapes, xyz_camera=joints,
                        rgb_spatial=features, crop_rotations=np.concatenate(crop_rotations),
                        crop_focals=np.asarray(crop_focals), right=np.asarray(rights))
    result = dict(frames=observations, count=len(observations), tracks=next_id, GT_used=False,
                  decoder='WristAnchoredMANO21; right-canonical local rotations plus explicit left reflection',
                  wrist_is_not_MANO_transl=True, raw_WiLoR_vs_SO3_FK_max_mm=maximum_parity*1000,
                  crop_recipe='Pinned Ego-Hand perspective fisheye crop, rotation1, BF16; not private 5090 pixel parity',
                  seconds=time.time()-started, helper_checkpoint=helper_path, track_side_decisions=decisions, handedness_policy='prediction-only confidence-weighted track votes BEFORE actual WiLoR crop inference')
    path = args.output/'observations.json'
    path.write_text(json.dumps(result, indent=2))
    hashes = {name: hashlib.file_digest((args.output/name).open('rb'), 'sha256').hexdigest()
              for name in ['observations.json', 'observations.npz']}
    (args.output/'seal.json').write_text(json.dumps(dict(files=hashes, GT_used=False), indent=2))
    print(json.dumps(dict(complete=True, observations=len(observations), tracks=next_id,
                          raw_WiLoR_vs_SO3_FK_max_mm=maximum_parity*1000)), flush=True)


if __name__ == '__main__':
    main()
