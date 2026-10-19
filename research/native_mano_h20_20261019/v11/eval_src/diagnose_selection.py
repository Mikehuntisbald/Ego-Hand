"""Only train/dev-select labels; inspect composite-score conservatism."""
import json,sys
from pathlib import Path
import torch
R=Path('/mnt/why/native_candidate_risk_v11_20261019');sys.path.insert(0,str(R/'src'))
from training_data import TrainingData
from risk_model import NativeQuality
def main():
    torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);data=TrainingData();y=data.labels;score=y['delta_error_mm'].mean(-1)/5;composite=score+2*y['harm_on_good'].float().mean(-1);report=dict(train_dev_only=True,GT_error_only_best_counts=torch.bincount(score.argmin(1),minlength=6).cpu().tolist(),GT_composite_best_counts=torch.bincount(composite.argmin(1),minlength=6).cpu().tolist(),arms={})
    b,c,target=data.get(data.select)
    for arm in ['pose_base','RGB_base','RGB_preserve']:
        ck=torch.load(R/'runs'/arm/'best_development.pt',map_location='cpu',weights_only=False);model=NativeQuality(use_rgb=arm!='pose_base').cuda().eval();model.load_state_dict(ck['model'])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(b,c)
        delta=out['delta_error_mm'].mean(-1)/5;report['arms'][arm]=dict(best_step=int(ck['selected_step']),dev_error_only_argmin_counts=torch.bincount(delta.argmin(1),minlength=6).cpu().tolist(),dev_composite_argmin_counts=torch.bincount(out['score'].argmin(1),minlength=6).cpu().tolist(),pred_delta_min=float(delta[:,1:].min()),pred_delta_mean=float(delta[:,1:].mean()),risk_penalty_mean=float((2*out['harm_probability'][:,1:].mean(-1)).mean()));del model
    (R/'receipts/score_diagnosis_train_dev.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
