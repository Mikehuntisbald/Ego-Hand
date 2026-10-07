import json
import numpy as np,torch
from hand3d_native_v10 import NativeTrajectoryHand3D
from hand3d_v8_common import V7,load,batch,save
from hand3d_trajectory_data_v9 import RUN as V9,targets
from hand3d_visual_v8 import dense_bank
from train_hand3d_v7 import initialize
RUN=V7.parent/'offline_hand3d_v10_native'

def main():
    torch.set_num_threads(4);device='cuda:0';data=load(device);bank=dense_bank(device,len(data['world']));extra={k:v.to(device) for k,v in torch.load(V9/'trajectory_targets.pt',weights_only=False).items()};ids=torch.tensor(np.where(np.asarray(data['roles'])=='train')[0][:4],device=device);prob=torch.load(V7/'risk_probabilities.pt',weights_only=False)['train_oof'].to(device);b=batch(data,ids,prob);b['rgb_native']=bank[data['feature_ids'][ids]];gt,valid,uv,uvvalid=targets(data,extra,ids);changed=gt.clone();changed[(~valid).all(-1)]+=100;reports={}
    for channels in [128,1280]:
        model=NativeTrajectoryHand3D('dit',channels==1280).to(device);initialize(model,device);model.initialize_visual(device);opt=torch.optim.AdamW(model.parameters(),lr=1e-4);model.eval()
        for step in range(2):
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,gt,valid,uv,uvvalid)
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        gradients=dict(native_projection=float(model.rgb_project.weight.grad.norm()),localization=float(model.localization.project[1].weight.grad.norm()));assert all(np.isfinite(v) and v>0 for v in gradients.values())
        values=[]
        with torch.no_grad():
            for target in [gt,changed]:
                torch.manual_seed(202610107)
                with torch.autocast('cuda',dtype=torch.bfloat16):values.append(float(model.loss(b,target,valid,uv,uvvalid)))
            assert values[0]==values[1],values
            with torch.autocast('cuda',dtype=torch.bfloat16):
                encoded=model.encode(b);x=torch.randn(4,17,21,3,device=device)*model.token_mask;t=torch.full((4,),50,device=device,dtype=torch.long);output=model.net(x,t,encoded);altered=x.clone();altered[~b['rgb_valid']]+=100;other=model.net(altered,t,encoded)
            center_delta=float((output[:,8]-other[:,8]).abs().max());assert center_delta==0,center_delta
        reports[str(channels)]=dict(passed=True,invalid_label_placeholder_loss_invariance=True,missing_frame_latent_center_invariance=True,invalid_labels_loss_delta=abs(values[0]-values[1]),missing_latent_center_delta=center_delta,gradient_norms=gradients)
        del model,opt;torch.cuda.empty_cache()
    save(RUN/'preflight.json',reports);save(RUN/'invalid_target_fix_evidence.json',dict(before=dict(loss_original=.7097337245941162,loss_perturbed=.7343343496322632,delta=.024600625038146973,invalid_frames=61),after=json.loads((RUN/'invalid_target_diagnosis.json').read_text()),inference='Previous v9 inference never consumed GT; bug was an invalid-label training target and padding problem'))
    print(json.dumps(reports),flush=True)

if __name__=='__main__':main()
