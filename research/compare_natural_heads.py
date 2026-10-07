"""Same-protocol comparison; select each arm only from saved development calibration."""
import json
import numpy as np,torch
import spatial_rgb_common as s
from natural_reliability import RUN
from natural_policy import apply_policy
from evaluate_natural_reliability import measure
from evaluate_offline_kp import bootstrap
torch.set_num_threads(4)
search=json.loads((RUN/'calibration_search.json').read_text());p=np.load(RUN/'predictions.npz')
base=torch.tensor(p['base']);gt=torch.tensor(p['gt']);available=torch.tensor(p['available']);m=torch.tensor(p['valid'])&available
candidates={a:torch.tensor(p[a]) for a in ['dit','regression','rgb_probe']};risk=torch.tensor(p['p_bad']);confirmed=torch.zeros_like(available)
_,index=s.records_and_index();ids=p['window_indices'];roi=index['roi'][index['feature_ids'][ids,8]]
rows=json.loads((s.OLD/'rows.json').read_text());clusters=np.array([rows[int(i)]['sequence'] for i in ids])
hard_ids={q['window_index'] for q in json.loads((s.RUN/'natural_finger_audit/candidates.json').read_text())};h=torch.tensor([int(i) in hard_ids for i in ids])
report=dict(scope='Post-hoc same-input audit on reused test data; each policy selected from saved development calibration only. No new fitting or threshold search.',methods={});errors={}
for arm in ['dit','regression']:
    approved=[c for c in search['configs'] if c['policy']['arm']==arm and c['passed']]
    if not approved:report['methods'][arm]=dict(calibration_passed=False);continue
    best=min(approved,key=lambda c:c['metrics']['mean_px'])
    out,selected=apply_policy(base,candidates,risk,available,confirmed,roi,best['policy'])
    report['methods'][arm]=dict(calibration_passed=True,policy=best['policy'],development=best['metrics'],all_test=measure(out,base,gt,m,selected),hardcases=measure(out[h],base[h],gt[h],m[h],selected[h]))
    errors[arm]=(out-gt).norm(dim=-1)*1408
if len(errors)==2:
    delta=errors['dit']-errors['regression']
    report['dit_minus_regression']=dict(all_mean_px=float(delta[m].mean()),all_ci95_px=bootstrap(delta.numpy(),m.numpy(),clusters),
        hardcase_mean_px=float(delta[h][m[h]].mean()),hardcase_ci95_px=bootstrap(delta[h].numpy(),m[h].numpy(),clusters[h]))
s.save(RUN/'head_comparison.json',report);print(json.dumps(report,indent=2))
