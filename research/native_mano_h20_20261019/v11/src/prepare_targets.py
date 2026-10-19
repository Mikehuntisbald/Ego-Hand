"""Native3D risk labels; never used by the deployment input wrapper."""
import hashlib,json
from pathlib import Path
import torch
R=Path('/mnt/why/native_candidate_risk_v11_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ready=json.loads((R/'data/candidates_ready.json').read_text());assert ready['complete'] and not ready['hand_GT_read'];assert sha(R/'data/candidates.pt')==ready['candidate_sha256'];assert not (R/'data/targets_ready.json').exists()
    bank=torch.load(R/'data/candidates.pt',map_location='cpu',weights_only=False);native=torch.load(V7/'data/native_targets.pt',map_location='cpu',weights_only=False);raw=torch.load(V7/'data/inputs.pt',map_location='cpu',weights_only=False);lookup={int(i):j for j,i in enumerate(bank['indices'])};keep=[j for j,i in enumerate(native['indices']) if int(i) in lookup];assert len(keep)==1355
    native_idx=native['indices'][keep];input_index=torch.tensor([lookup[int(i)] for i in native_idx]);GT=native['gt'][keep];points=bank['joints_camera21'][input_index];p=points-points[:,:,:1];g=GT-GT[:,:1];error=(p-g[:,None]).norm(dim=-1)[:,:,1:]*1000;delta=error-error[:,:1];good=error[:,0]<=10;harm=(delta>1)&good[:,None];distance=(p-p[:,:1]).norm(dim=-1)[:,:,1:]*1000;assert bool((delta.abs()<=distance+2e-4).all()),'Triangle bound failed'
    roles=[raw['rows'][int(i)]['role'] for i in native_idx];assert roles.count('train')==1330 and roles.count('dev_select')==25
    out=dict(indices=native_idx,input_index=input_index,error_mm=error,delta_error_mm=delta,reference_good=good,harm_on_good=harm,GT_camera21=GT);target=R/'data/quality_targets.pt';torch.save(out,target)
    gt_score=delta.mean(-1)/5+2*harm.float().mean(-1);best=gt_score.argmin(1);report=dict(complete=True,train=1330,dev_select=25,targets_sha256=sha(target),frozen_candidate_sha256=ready['candidate_sha256'],source_native_targets_sha256=sha(V7/'data/native_targets.pt'),native_error_difference_triangle_bound=True,reference_good_points=int(good.sum()),harmful_events=int(harm.sum()),GT_best_composite_candidate_counts=torch.bincount(best,minlength=6).tolist(),GT_labels_loss_only=True,production_admitted=False)
    (R/'data/targets_ready.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
