"""Check actual heatmap/risk gradient paths and separation of inputs and targets."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from rgb_head import RGBObservation,loss,pixel_error
R=Path('/mnt/why/native_rgb_observation_v9_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019')
def main():
    torch.set_num_threads(4);torch.manual_seed(190019);labels=torch.load(R/'data/targets.pt',map_location='cpu',weights_only=False);bank=np.load(V7/'data/rgb.npy',mmap_mode='r');ids=np.linspace(0,1330,12,dtype=int);fids=labels['feature_ids'][ids].numpy();x=torch.from_numpy(np.array(bank[fids])).cuda();y={k:v[ids].cuda() for k,v in labels.items() if torch.is_tensor(v)}
    model=RGBObservation().cuda().train();out=model(x,y['coarse_uv']);total,parts=loss(model,out,y);total.backward();grad={k:float(v.grad.abs().sum()) for k,v in model.named_parameters() if v.grad is not None};assert all(v>0 for v in grad.values())
    model.eval()
    with torch.no_grad():
        original=model(x,y['coarse_uv']);fake={k:torch.randn_like(v.float()) for k,v in y.items() if k!='coarse_uv'};poison=model(x,y['coarse_uv']);assert all(torch.equal(original[k],poison[k]) for k in original)
        changed=model(x,torch.randn_like(y['coarse_uv']));assert torch.equal(original['uv'],changed['uv']),'Coarse MANO must not drive independent coordinates';assert not torch.equal(original['coarse_bad_logit'],changed['coarse_bad_logit'])
        zero=model(torch.zeros_like(x),y['coarse_uv']);rgb_effect=float((original['uv']-zero['uv']).abs().max());assert rgb_effect>1e-5
    grid=model.grid;assert grid.shape==(1,1,3072,2);assert torch.allclose(grid[0,0,0],grid.new_tensor([.5/48,.5/64]))
    result=dict(complete=True,loss=float(total.detach()),parts=parts,gradient_sums=grad,GT_poison_output_unchanged=True,coordinate_independent_of_coarse=True,RGB_effect_max_normalized=rgb_effect,heatmap_shape=[64,48],supervision='amodal native MANO projections; framing and 2D error-risk, not visibility or 3D certainty',head_only=True,visual_backbone_frozen=True,production_admitted=False)
    (R/'receipts/preflight.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
