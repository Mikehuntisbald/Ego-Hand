"""Development sampling audit, never uses locked sequences."""
import json
import time
import wilor_eval_common
import torch
from train_dit_v3 import RUN,load_data,batch,save
from dit_v3_model import VisualResidual
from dit_v3_sampling import propose
from metrics_3d import EVAL_INDICES

folder=RUN/'initial_dit'
while not (folder/'proposal_done.json').exists():time.sleep(10)
torch.set_num_threads(4)
rows,data=load_data('cuda:0')
ck=torch.load(folder/'proposal_best.pt',map_location='cuda:0',weights_only=False)
model=VisualResidual('dit').to('cuda:0').eval();model.load_state_dict(ck['model'])
ids=torch.tensor(json.loads((folder/'training_config.json').read_text())['selection_indices'],device='cuda:0')
gt=data['gt'][ids];coarse=data['coarse'][ids]
before=(coarse-gt).norm(dim=-1)[:,EVAL_INDICES]*1000
rb=((coarse-coarse[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)[:,EVAL_INDICES]*1000
results=[]
with torch.inference_mode():
    for policy,steps,samples in [('independent',10,2),('antithetic',10,2),('antithetic',10,4),('zero',10,1),('zero',20,1),('independent',10,8)]:
        generator=torch.Generator(device='cuda:0').manual_seed(914003);pred=[];start_time=time.time()
        for start in range(0,len(ids),64):
            b=batch(data,ids[start:start+64])
            with torch.autocast('cuda',dtype=torch.bfloat16):p=propose(model,b,policy=policy,steps=steps,samples=samples,generator=generator)
            pred.append(b['coarse']+model.native_delta(p,b))
        pred=torch.cat(pred)
        camera=(pred-gt).norm(dim=-1)[:,EVAL_INDICES]*1000
        relative=((pred-pred[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)[:,EVAL_INDICES]*1000
        row=dict(policy=policy,steps=steps,samples=samples,camera_mm=float(camera.mean()),relative_mm=float(relative.mean()),
                 harm_camera=float(((camera>before+1)&(before<=10)).sum()/(before<=10).sum().clamp_min(1)),
                 harm_relative=float(((relative>rb+1)&(rb<=10)).sum()/(rb<=10).sum().clamp_min(1)),seconds=time.time()-start_time)
        results.append(row);print(json.dumps(row),flush=True)
save(folder/'sampling_development.json',dict(proposal_step=ck['step'],development_indices=ids.cpu().tolist(),results=results,locked_data_used=False))
