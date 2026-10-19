import json
from pathlib import Path
R=Path('/mnt/why/native_candidate_risk_v11_20261019')
def main():
    for arm in ['pose_base','RGB_base','RGB_preserve']:
        p=R/'runs'/arm/'progress.json'
        if not p.exists():continue
        v=json.loads(p.read_text());s=v['selection'];approved={k:s[k] for k in ['native_relative20_mm','reference_mm','accurate_points','harmed_over1mm','hard_points','recovered_to20','delta_MAE_nonreference_mm','harm_Brier_nonreference']};assert all(isinstance(x,(int,float)) for x in approved.values());print(json.dumps(dict(arm=arm,step=int(v['step']),metrics=approved)))
    v=json.loads((R/'receipts/actual_resume.json').read_text());print(json.dumps(dict(actual_resume=bool(v['complete']) and all(v['exact_fields'].values()))))
if __name__=='__main__':main()
