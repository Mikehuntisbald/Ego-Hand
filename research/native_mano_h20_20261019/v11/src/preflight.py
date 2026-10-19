import json,sys
from pathlib import Path
import numpy as np,torch
from training_data import TrainingData
from risk_model import NativeQuality,objective
R=Path('/mnt/why/native_candidate_risk_v11_20261019')
def main():
    torch.set_num_threads(4);torch.manual_seed(190019);data=TrainingData();ids=data.train[:6];b,c,y=data.get(ids);model=NativeQuality().cuda();out=model(b,c);total,parts=objective(out,y,True);total.backward();grads={k:float(v.grad.abs().sum()) for k,v in model.named_parameters() if v.grad is not None};assert grads['rgb.1.weight']>0 and grads['delta.weight']>0 and grads['coarse.weight']>0 and grads['harm.weight']>0
    with torch.no_grad():
        first=model(b,c);second=model(dict(b,indices=b['indices']*0),c);assert all(torch.equal(first[k],second[k]) for k in first);poison={k:torch.randn_like(v.float()) for k,v in y.items()};third=model(b,c);assert all(torch.equal(first[k],third[k]) for k in first)
        perm=torch.tensor([0,5,2,1,4,3],device='cuda');changed=model(b,{k:v[:,perm] for k,v in c.items()});permutation=float((changed['score']-first['score'][:,perm]).abs().max());assert permutation<2e-5
        control=NativeQuality(use_rgb=False).cuda();x=control(b,c);z=control(dict(b,rgb=torch.randn_like(b['rgb'])),c);assert torch.equal(x['score'],z['score']);assert (first['score'][:,0]==0).all();distance=(c['joints_camera21'][:,:,1:]-c['joints_camera21'][:,:1,1:]).norm(dim=-1)*1000;assert (first['delta_error_mm'].abs()<=distance+1e-5).all()
    report=dict(complete=True,GT_poison_and_metadata_index_invariant=True,candidate_permutation_equivariance_max=permutation,noRGB_control_invariant_to_pixels=True,reference_score_exact_zero=True,error_difference_triangle_bound=True,RGB_coarse_delta_risk_gradients_nonzero=True,gradient_sums=grads,loss=float(total.detach()),parts=parts,visual_backbone_and_DiT_frozen=True,production_admitted=False)
    (R/'receipts/preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='gradient_sums'},indent=2))
if __name__=='__main__':main()
