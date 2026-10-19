import json,sys
from pathlib import Path
import torch
R=Path('/mnt/why/native_candidate_risk_v11_20261019/error_only_r1');sys.path.append('/mnt/why/native_candidate_risk_v11_20261019/src');sys.path.insert(0,str(R/'src'))
from training_data import TrainingData
from risk_model_r1 import NativeQuality,objective
def main():
    torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.manual_seed(190019);d=TrainingData();b,c,y=d.get(d.train[:6]);m=NativeQuality().cuda();opt=torch.optim.AdamW(m.parameters(),lr=2e-4)
    for _ in range(3):
        opt.zero_grad();out=m(b,c);loss,_=objective(out,y,True);loss.backward();opt.step()
    m.zero_grad();out=m(b,c);out['score'][:,1:].sum().backward();pixel_gradient=float(m.rgb[1].weight.grad.abs().sum());assert pixel_gradient>0
    with torch.no_grad():
        original=m(b,c);ignored=m(dict(b,indices=b['indices']*0),c);assert torch.equal(original['score'],ignored['score']);assert (original['score'][:,0]==0).all();perm=torch.tensor([0,5,2,1,4,3],device='cuda');permuted=m(b,{k:v[:,perm] for k,v in c.items()});gap=float((original['score'][:,perm]-permuted['score']).abs().max());assert gap<2e-5;zero=m(dict(b,rgb=torch.zeros_like(b['rgb'])),c);RGB_effect=float((zero['score']-original['score']).abs().max());assert RGB_effect>1e-6
    report=dict(complete=True,post_update_error_score_to_RGB_gradient=pixel_gradient,RGB_score_effect=RGB_effect,candidate_permutation_max=gap,metadata_invariant=True,reference_score_exact_zero=True,source_targets_used_only_in_loss=True,production_admitted=False);(R/'receipts/preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
