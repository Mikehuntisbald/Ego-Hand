import json,time
import numpy as np,torch
from hand3d_data_v7 import RUN as V7,load,batch,save
from hand3d_rollout_v8 import RolloutHand3D,project_fisheye624
from train_hand3d_v7 import metrics
import spatial_rgb_common as s
RUN=V7.parent/'offline_hand3d_v8';RUN.mkdir(exist_ok=True)

def camera_bank():
    path=RUN/'camera_params.pt'
    if path.exists():return torch.load(path,weights_only=False)
    records,_=s.records_and_index();params=torch.zeros(len(records)+1,16)
    for i,r in enumerate(records,1):
        cam=s.common.from_json(r['camera']);params[i]=torch.tensor(list(cam.f)+list(cam.c)+list(cam.distort))
    torch.save(params,path);return params

def initialize(kind,device):
    model=RolloutHand3D(kind,True).to(device)
    initial=torch.load(V7/kind_arm(kind)/'best.pt',weights_only=False,map_location=device)
    result=model.load_state_dict(initial['model'],strict=False)
    assert all(k.startswith('coherent_trust.') for k in result.missing_keys) and not result.unexpected_keys
    return model,initial['step']

def kind_arm(kind):return 'rgb_'+kind

@torch.no_grad()
def predict(model,data,prob,ids,size=16,visual=None):
    model.eval();out={}
    if visual is not None:visual.eval()
    for start in range(0,len(ids),size):
        ix=ids[start:start+size];b=batch(data,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if visual is not None:b=visual.replace_batch(b,data,ix)
            p=model.predict(b,seed=202610081+start)
        for key in ['xyz_camera_m','raw_xyz_camera_m','std_m','trust']:
            out.setdefault(key,[]).append(p[key].float())
    return {k:torch.cat(v) for k,v in out.items()}

def score(result):
    improvement=(result['base_relative_mm']-result['relative_mm'])/result['base_relative_mm']
    feasible=(result['camera_good_harm_rate']<=.01 and result['relative_good_harm_rate']<=.01 and result['camera_mm']<=result['base_camera_mm']+.1 and improvement>=.05)
    penalty=max(0,result['camera_good_harm_rate']-.01)*200+max(0,result['relative_good_harm_rate']-.01)*200+max(0,result['camera_mm']-result['base_camera_mm']-.1)*2
    return (0 if feasible else 1,penalty+result['camera_mm']+.75*result['relative_mm']),feasible

def main_preflight():
    torch.set_num_threads(4);device='cuda:0';data=load(device);params=camera_bank().to(device)
    records,_=s.records_and_index();errors=[]
    for i in range(1,len(records)+1,37):
        xyz=data['xyz_camera_bank'][i].float();cam=s.common.from_json(records[i-1]['camera'])
        with torch.no_grad():uv=project_fisheye624(xyz[None],params[i:i+1])[0].cpu().numpy()
        expected=cam.eye_to_window(xyz.cpu().numpy());errors.append(float(np.max(np.abs(uv-expected))))
    parity=max(errors);assert parity<.01,parity
    roles=np.asarray(data['roles']);train=torch.tensor(np.where(roles=='train')[0],device=device)
    risk=torch.load(V7/'risk_probabilities.pt',weights_only=False)['train_oof'].to(device)
    model,initial=initialize('dit',device);model.eval();ix=train[:8];b=batch(data,ix,risk)
    started=time.time()
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.loss(b,data['gt'][ix],data['valid'][ix],data['gt_uv'][ix],data['uv_valid'][ix],params[data['feature_ids'][ix,8]],812)
    assert torch.isfinite(loss);loss.backward();norms={k:float(p.grad.norm()) for k,p in model.named_parameters() if k in ['head.weight','rgb_project.weight','coherent_trust.2.weight']}
    assert all(np.isfinite(v) and v>0 for v in norms.values()),norms
    model.zero_grad(set_to_none=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        p=model.predict(b);locks=torch.zeros(8,20,device=device,dtype=torch.bool);locks[:,[0,5,12]]=True
        locked=batch(data,ix,risk,locks);pl=model.predict(locked)
        assert torch.equal(pl['xyz_camera_m'][locks],b['base'][locks])
        poisoned={**b,'gt':torch.randn_like(data['gt'][ix])*10,'visibility':torch.rand(8,20,device=device),'gt_handedness':torch.ones(8,device=device)}
        pp=model.predict(poisoned);leak=float((p['xyz_camera_m']-pp['xyz_camera_m']).abs().max());assert leak==0
    result=dict(complete=True,projection_max_px=parity,gradient_norms=norms,output_shape=list(p['xyz_camera_m'].shape),locks_exact=True,gt_poison_max_m=leak,batch=8,seconds=time.time()-started,max_cuda_gb=torch.cuda.max_memory_allocated()/1e9,initial_v7_step=initial,loss=float(loss),parts=parts)
    save(RUN/'preflight.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main_preflight()
