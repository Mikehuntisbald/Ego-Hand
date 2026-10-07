"""Freeze v3 data roles and new source-sequence evaluation before training."""
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT=Path('/mnt/why/HOT3D')
RUN=ROOT/'experiments/dit_wilor_v3'
RUN.mkdir(exist_ok=True)
old=json.loads((ROOT/'subset_manifest.json').read_text())
v2=json.loads((ROOT/'experiments/dit_subject_oof_v2/protocol.json').read_text())
used={s['sequence'] for s in old['sequences']}|{s['sequence'] for s in v2['fresh_sequences']}
defs=json.loads((ROOT/'provenance/clip_definitions.response').read_text())
splits=json.loads((ROOT/'provenance/clip_splits.response').read_text())
available=defaultdict(list)
for clip in splits['train']['Aria']:
    seq=defs[str(clip)]['sequence_id']
    if seq not in used:available[seq].append(int(clip))
rng=random.Random(202610031)
fresh=[]
for subject in ['P0003','P0010','P0015']:
    seqs=sorted(s for s,clips in available.items() if s.startswith(subject+'_') and len(clips)>=6)
    rng.shuffle(seqs)
    for seq in seqs[:4]:
        clips=available[seq].copy();rng.shuffle(clips)
        fresh.append(dict(split='dit_v3_locked',subject=subject,sequence=seq,clips=sorted(clips[:6])))
protocol=dict(study='dit_wilor_v3',seed=202610031,
    train_subjects=v2['subjects'],denoise_sequences=v2['denoiser_sequences'],gate_sequences=v2['gate_fit_sequences'],
    development='All previously evaluated sequences of P0003/P0010/P0015; no longer independent tests',
    locked_sequences=fresh,locked_scope='Previously unused source sequences; same development subjects, not wholly new subjects; WiLoR upstream HOT3D overlap unknown',
    baseline='Frozen original YOLO26 coarse estimator; OOF predictions for training, final frozen model for deployment',
    conditioning='Frozen full WiLoR trained visual features, joint-local evidence, predicted pose and side, calibrated camera crop geometry, coarse uncertainty; no true side/visibility/pose/shape as input',
    geometry='Numerical inversion of calibrated fisheye without one-radian clamp; rotation and handedness applied consistently to features and 3D targets',
    models=['conditional_residual_dit','matched_conditional_regression'],
    acceptance=dict(hard_groups=['low_pose_confidence','high_occlusion_in_view','multiple_ray_aligned_fingers'],
        minimum_relative_improvement_percent=5,paired_source_sequence_ci95_improvement_lower_gt=0,
        initially_correct_threshold_mm=10,meaningful_harm_mm=1,maximum_correct_point_harm_fraction=.05,
        protection_spaces=['camera','wrist_relative'],overall_camera_regression_tolerance_mm=.1,
        minimum_group_joints=100),
    operating_policy='Fit continuous local gate on held training sequences; choose operating point on development only. Do not unlock new evaluation until all development acceptance checks pass.',
    scientific_scope='Deployment improvement over fixed baseline. Diffusion-specific superiority must be checked against identical-feature regression; no unsupported claim.',
    status='preparing_condition_cache')
dest=RUN/'protocol.json'
if dest.exists():assert json.loads(dest.read_text())==protocol,'Never overwrite an established protocol'
else:dest.write_text(json.dumps(protocol,indent=2))
(RUN/'locked_manifest.json').write_text(json.dumps({**{k:old[k] for k in ['official_repo','revision','mirror','stream']},'sequences':fresh},indent=2))
(RUN/'status.json').write_text(json.dumps(dict(stage='condition_cache',complete=False,acceptance_passed=False),indent=2))
print(json.dumps(dict(run=str(RUN),train_sequences=len(protocol['denoise_sequences']),gate_sequences=len(protocol['gate_sequences']),locked_sequences=len(fresh),locked_clips=sum(len(s['clips']) for s in fresh)),indent=2))
