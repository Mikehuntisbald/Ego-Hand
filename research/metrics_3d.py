"""Compare baseline/refined camera-space hand predictions; meters in, mm out."""
import numpy as np
EVAL_INDICES=np.array([0,1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19])

def pose_metrics(pred,gt):
    pred=np.asarray(pred,dtype=np.float64);gt=np.asarray(gt,dtype=np.float64)
    assert pred.shape==gt.shape and pred.ndim==3 and pred.shape[1:]==(20,3)
    assert np.isfinite(pred).all() and np.isfinite(gt).all()
    errors=np.linalg.norm(pred[:,EVAL_INDICES]-gt[:,EVAL_INDICES],axis=-1)*1000
    root_error=np.linalg.norm(pred[:,5]-gt[:,5],axis=-1)*1000
    relative_pred=pred-pred[:,5:6];relative_gt=gt-gt[:,5:6]
    relative_error=np.linalg.norm(relative_pred[:,EVAL_INDICES]-relative_gt[:,EVAL_INDICES],axis=-1)*1000
    return dict(mpjpe19_mm=float(errors.mean()),wrist_relative_mpjpe19_mm=float(relative_error.mean()),
                root_translation_mm=float(root_error.mean()),
                pck={str(t):float((errors<=t).mean()) for t in (5,10,20)},
                sample_mpjpe19_mm=errors.mean(axis=1).tolist())

def compare(coarse,refined,gt,correct_threshold_mm=10,harm_margin_mm=1):
    initial=pose_metrics(coarse,gt);final=pose_metrics(refined,gt)
    before=np.linalg.norm(np.asarray(coarse)[:,EVAL_INDICES]-np.asarray(gt)[:,EVAL_INDICES],axis=-1)*1000
    after=np.linalg.norm(np.asarray(refined)[:,EVAL_INDICES]-np.asarray(gt)[:,EVAL_INDICES],axis=-1)*1000
    correct=before<=correct_threshold_mm;harm=after>before+harm_margin_mm
    return dict(coarse=initial,refined=final,mpjpe19_change_mm=final['mpjpe19_mm']-initial['mpjpe19_mm'],
                initially_correct_joints=int(correct.sum()),
                correct_joints_harmed_fraction=float((correct&harm).sum()/max(1,correct.sum())),
                harmed_joint_fraction=float(harm.mean()),
                improved_joint_fraction=float((after+harm_margin_mm<before).mean()))
