import os,json
from pathlib import Path
run=Path(os.environ.get('HOT3D_DIT_RUN','/mnt/why/HOT3D/experiments/dit_wilor_v5'))
p=run/'locked_results.json'
if not p.exists():
    for name in ['status.json','locked_data_status.json']:
        if (run/name).exists():print((run/name).read_text())
else:
    r=json.loads(p.read_text());out={k:r[k] for k in ['accepted','sampled_frames','matched_hands','subjects','scope']}
    out['methods']={}
    for kind,m in r['methods'].items():
        a=m['overall'];out['methods'][kind]=dict(accepted=m['accepted'],camera=[a['coarse']['mpjpe19_mm'],a['refined']['mpjpe19_mm']],
            relative=[a['coarse']['wrist_relative_mpjpe19_mm'],a['refined']['wrist_relative_mpjpe19_mm']],
            harm_camera=a['correct_joints_harmed_fraction'],harm_relative=m['groups']['all']['correct_relative_joints_harmed_fraction'],
            evidence=m['evidence'],held_subject_no_regression=m.get('held_subject_no_regression'),
            held_relative=[m['per_subject']['P0015'][k]['wrist_relative_mpjpe19_mm'] for k in ['coarse','refined']] if 'per_subject' in m else None)
    out['dit_vs_regression']={k:g.get('axial_relative' if k=='multiple_ray_aligned_fingers' else 'relative') for k,g in r['dit_vs_regression'].items()}
    print(json.dumps(out,indent=2))
