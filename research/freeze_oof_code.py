"""Record actual code/environment before seeing final sequence results."""
import torch,ultralytics
from oof_common import RUN,PROJECT,save,sha
files=['coarse_pose3d.py','residual_models.py','pose_residual_dit.py','train_oof_coarse.py','cache_oof_predictions.py',
    'train_oof_residual.py','oof_calibration.py','evaluate_oof_study.py','infer_oof_pose.py']
save(RUN/'code_manifest.json',dict(torch=torch.__version__,ultralytics=ultralytics.__version__,files={f:sha(PROJECT/f) for f in files},
    probability_calibration_objective='camera MPJPE + wrist-relative MPJPE + 0.5 low calibrated confidence wrist-relative MPJPE',
    gate_calibration_scope='Independent of direct residual/gate training and downstream checkpoint selection; upstream coarse selection has used P0003',
    data_extension='Official immutable v1 dataset revision; six new original sequences; RGB/calibration/hand GT only',
    no_fresh_or_development_results_used_for_v2_selection=True))
print((RUN/'code_manifest.json').read_text(),flush=True)

