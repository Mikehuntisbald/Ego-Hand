"""Check all already-calibrated operating points for the full acceptance rule."""
import json
import argparse
import wilor_eval_common
import torch
from train_dit_v3 import RUN,load_data,save
from dit_v3_model import VisualResidual
from gate_dit_v3 import masks
from evaluate_proof import group_report

torch.set_num_threads(4)
parser=argparse.ArgumentParser()
parser.add_argument('--arms',nargs='+',default=['initial_dit','zero_dit','antithetic_dit','ray_focus_dit','ray_focus_regression'])
args=parser.parse_args()
rows,data=load_data('cuda:0');ids=torch.tensor([i for i,r in enumerate(rows) if r['role']=='development'],device='cuda:0')
groups,clusters=masks(rows,data,ids);c=data['coarse'][ids].cpu().numpy();gt=data['gt'][ids].cpu().numpy()
for name in args.arms:
    folder=RUN/name
    if not (folder/'gate_done.json').exists():continue
    result=json.loads((folder/'development_results.json').read_text());grid=json.loads((folder/'calibration_grid.json').read_text())
    candidates=[]
    for row in grid:
        if not row['admissible']:continue
        if all(row['group_errors'][g]<=result['groups'][g]['axial_relative' if g=='multiple_ray_aligned_fingers' else 'relative']['before_mm']*.95 for g in row['group_errors']):candidates.append(row)
    ck=torch.load(folder/'best.pt',map_location='cuda:0',weights_only=False);model=VisualResidual(ck['kind']).to('cuda:0').eval();model.load_state_dict(ck['model'])
    bank=torch.load(folder/'gate_banks.pt',map_location='cuda:0',weights_only=False)['dev']
    reports=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):raw=model.gate(bank['inputs'].float()).squeeze(-1).sigmoid().float()
    for op in candidates:
        gates=raw*(raw>=op['threshold'])*op['strength']
        pred=(data['coarse'][ids]+gates[...,None]*bank['delta']).cpu().numpy()
        g={k:group_report(pred,gt,c,m.cpu().numpy(),clusters) for k,m in groups.items() if k!='all'}
        accepted=all(r['joints']>=100 and r['axial_relative' if k=='multiple_ray_aligned_fingers' else 'relative']['ci95'][1]<0 for k,r in g.items())
        reports.append(dict(operating=op,groups=g,accepted=accepted))
    valid=[r for r in reports if r['accepted']]
    save(folder/'operating_acceptance_scan.json',dict(candidates=reports,any_accepted=bool(valid)))
    if valid:
        chosen=min(valid,key=lambda r:r['operating']['score']);ck['operating']=chosen['operating'];torch.save(ck,folder/'accepted_operating.pt');save(folder/'accepted_development.json',chosen)
    print(json.dumps(dict(arm=name,mean_eligible=len(candidates),accepted=len(valid))),flush=True)
    del bank,model;torch.cuda.empty_cache()
