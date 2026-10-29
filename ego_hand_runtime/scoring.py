"""Separate, post-seal aligned MANO21 GT evaluation. Never imported by inference."""
from pathlib import Path
import numpy as np
from .io import verify_seal, read_json, atomic_json, sha256
from .errors import RuntimeFailure


def score(prediction_dir, labels, output):
    # Freeze/verify predictions BEFORE the first label-file read.
    receipt = verify_seal(prediction_dir)
    prediction = read_json(Path(prediction_dir)/'joints.json')
    reference = read_json(labels)
    if reference.get('schema')!='aligned-MANO21-world-m-v1':
        raise RuntimeFailure('scoring_protocol', 'Require explicit MANO21 world-m aligned protocol; no implicit 20/21-point conversion')
    mapping = {(f['timestamp_ns'], str(h['track_id'])):h for f in prediction['frames'] for h in f['hands'] if h['valid']}
    errors, matched, eligible = [], 0, 0
    for frame in reference['frames']:
        for hand in frame['hands']:
            if not hand['valid']:
                continue
            eligible += 1
            pred = mapping.get((frame['timestamp_ns'], str(hand['track_id'])))
            if pred is None:
                continue
            if pred['right']!=hand['right']:
                raise RuntimeFailure('scoring_identity', 'Label and prediction chirality disagree')
            valid = np.asarray(hand.get('point_valid', [True]*21), bool)
            xyz = np.asarray(hand['joints_world_m'], float)
            if xyz.shape!=(21,3) or valid.shape!=(21,) or not np.isfinite(xyz[valid]).all():
                raise RuntimeFailure('scoring_labels', 'Invalid GT geometry/mask')
            errors.extend(np.linalg.norm(np.asarray(pred['joints_world_m'])-xyz, axis=-1)[valid].tolist())
            matched += 1
    if not errors:
        raise RuntimeFailure('scoring_empty', 'No eligible aligned GT/prediction points')
    e = np.asarray(errors)*1000
    result = dict(protocol=reference['schema'], points=len(e), matched_hand_frames=matched,
                  eligible_hand_frames=eligible, coverage=matched/max(eligible,1), MPJPE_mm=float(e.mean()),
                  RMSE_mm=float(np.sqrt(np.mean(e**2))), P95_mm=float(np.percentile(e,95)), maximum_mm=float(e.max()),
                  seal_verified_before_labels=True, labels_sha256=sha256(labels), prediction_binding=receipt['binding'],
                  absolute_accuracy_evidence=True, production_admission=False)
    atomic_json(output, result)
    return result
