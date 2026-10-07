"""Record verified completion and deployment-only changes after result lock."""
import json
from oof_common import RUN,PROJECT,save,sha
assert json.loads((RUN/'status.json').read_text())['complete']
assert json.loads((RUN/'inference_smoke.json').read_text())['passed']
save(RUN/'deployment_manifest.json',dict(files={f:sha(PROJECT/f) for f in ['infer_oof_pose.py','smoke_oof_inference.py']},
    weights_and_gate_calibration_changed_after_evaluation=False,final_test_repeated=False,
    corrections=['Match cached BF16 batch shape to avoid batch-size-dependent coarse error',
        'Match cached FP16 RGB token quantization'],pre_lock_code_manifest=str(RUN/'code_manifest.json')))
save(RUN/'status.json',dict(stage='complete',complete=True,positive_evidence=False,
    residual_and_gate_training_complete=True,inference_verified=True,
    result='Preservation passed, but OOF residuals do not improve hard-group recovery',
    report=str(RUN/'report.html'),results=str(RUN/'final_results.json')))
(RUN/'README.txt').write_text('''HOT3D subject-out-of-fold residual DiT study — 2026-10-03

Six coarse models were trained from COCO, excluding one complete training subject.
Residual training: 17,534 samples across 19 sequences; gates: 6,103 samples across 6 disjoint sequences.
Each refiner trained 5,000 proposal steps + 2,500 gate steps. Both direct regression and in-subject DiT controls are retained.
P0003 is development/selection/calibration only; its calibration sequence was held from downstream checkpoint selection but used for upstream coarse selection.
New evaluation: 24 clips, 6 new source sequences, 1,316 matched hands. Subjects P0010/P0015 are reused, not wholly new subjects.
No weights or operating points were changed after final evaluation. The inference interface was corrected to reproduce the cached numerical inputs.

Outcome: correct-point protection passes, but hard-group recovery does not improve reliably. Read report.html and final_results.json.
Weights and all training histories remain under this study directory. checkpoints.json provides full SHA256 values.
Python API: /mnt/why/hot3d_hand_residual/infer_oof_pose.py, SubjectOOFPoseRefiner.
Input: RGB hand crops normalized to [0,1], N x 3 x 256 x 256, and observed calibration geometry N x 21 from crop_and_geometry.
No GT, subject shape, true visibility mask or true pose is required by inference.
RGB encoder and coarse observation batches are padded to 256 to match study precision; residual timing excludes observation backbones and detection.

Resumable study pipeline: /mnt/why/HOT3D/.venv/bin/python /mnt/why/hot3d_hand_residual/run_oof_pipeline.py
Final evaluation is guarded against rerunning after tuning. New future studies need separate directories and their own evaluation protocol.
''')
print((RUN/'status.json').read_text(),flush=True)
