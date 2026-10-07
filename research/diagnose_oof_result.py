"""Read-only post-lock diagnostics; no parameters or operating points changed."""
import json,numpy as np
from metrics_3d import pose_metrics,EVAL_INDICES
from oof_common import RUN,save
result={}
for method in ['oof_dit','oof_regression','in_subject_dit','v1_dit']:
    data=np.load(RUN/f'fresh_{method}_predictions.npz')
    coarse=data['coarse'];gt=data['gt'];p=data['proposal'];full=coarse+p[:,:1]*.1+p[:,1:]*.03
    before=np.linalg.norm(coarse-gt,axis=-1)[:,EVAL_INDICES]
    after=np.linalg.norm(full-gt,axis=-1)[:,EVAL_INDICES]
    metrics=pose_metrics(full,gt);metrics.pop('sample_mpjpe19_mm')
    result[method]=dict(ungated=metrics,full_proposal_points_helpful_fraction=float((after+.001<before).mean()),
        full_proposal_points_harmful_fraction=float((after>before+.001).mean()),
        mean_applied_point_gate=float(data['gates'][:,1:][:,EVAL_INDICES].mean()),
        mean_full_point_update_mm=float(np.linalg.norm(full-coarse,axis=-1)[:,EVAL_INDICES].mean()*1000))
save(RUN/'post_lock_diagnosis.json',dict(read_only=True,no_new_parameter_selection=True,methods=result))
print((RUN/'post_lock_diagnosis.json').read_text(),flush=True)

