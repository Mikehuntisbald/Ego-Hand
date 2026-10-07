import argparse,json,time
import numpy as np,torch
from hand3d_data_v7 import RUN,V4,load,batch,save
from hand3d_temporal_v7 import WRIST
from hand3d_protected_v7 import ProtectedHand3D as TemporalHand3D
BASE_RUN=RUN
RUN=RUN.parent/'offline_hand3d_v7_protected'
RUN.mkdir(exist_ok=True)

def metrics(pred,base,gt,valid):
    mask=valid.clone();mask[:,WRIST]=False
    e=(pred-gt).norm(dim=-1)*1000;be=(base-gt).norm(dim=-1)*1000
    pe=(pred-pred[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1]);ce=(base-base[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])
    re=pe.norm(dim=-1)*1000;rbe=ce.norm(dim=-1)*1000
    result=dict(points=int(mask.sum()),camera_mm=float(e[mask].mean()),relative_mm=float(re[mask].mean()),wrist_mm=float(e[:,WRIST][valid[:,WRIST]].mean()),
        base_camera_mm=float(be[mask].mean()),base_relative_mm=float(rbe[mask].mean()),base_wrist_mm=float(be[:,WRIST][valid[:,WRIST]].mean()),pck20_camera=float((e[mask]<=20).float().mean()),pck20_relative=float((re[mask]<=20).float().mean()))
    for name,err,old in [('camera',e,be),('relative',re,rbe)]:
        bad=mask&(old>20);good=mask&(old<=10)
        result.update({name+'_bad_points':int(bad.sum()),name+'_bad_mean_mm':float(err[bad].mean()) if bad.any() else None,name+'_bad_recovered20':int((bad&(err<=20)).sum()),
            name+'_good_points':int(good.sum()),name+'_good_harmed20':int((good&(err>20)).sum()),name+'_good_harm_rate':float((err[good]>20).float().mean()) if good.any() else 0.})
    return result

def initialize(model,device):
    ck=torch.load(V4/'sealed/regression.pt',weights_only=False,map_location=device)['model']
    visual={k.removeprefix('visual_head.'):v for k,v in ck.items() if k.startswith('visual_head.')}
    model.visual_head.load_state_dict(visual)

@torch.no_grad()
def proposals(model,data,prob,ids,size=32):
    model.eval();out=[];std=[]
    for start in range(0,len(ids),size):
        ix=ids[start:start+size];b=batch(data,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict(b)
        out.append(p['xyz_camera_m'].float());std.append(p['std_m'].float())
    return torch.cat(out),torch.cat(std)

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['regression','dit'],required=True);p.add_argument('--tracks-only',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=4000);a=p.parse_args()
    torch.set_num_threads(4);seed=202610073;torch.manual_seed(seed);np.random.seed(seed)
    data=load(a.device);roles=np.array(data['roles']);risk=torch.load(BASE_RUN/'risk_probabilities.pt',weights_only=False);prob=risk['train_oof'].to(a.device);ep=risk['joint'].to(a.device)
    train=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=a.device)
    arm=('tracks_' if a.tracks_only else 'rgb_')+a.kind;folder=RUN/arm;folder.mkdir(exist_ok=True)
    model=TemporalHand3D(a.kind,not a.tracks_only).to(a.device);initialize(model,a.device)
    initial=torch.load(BASE_RUN/arm/'best.pt',weights_only=False,map_location=a.device);miss=model.load_state_dict(initial['model'],strict=False)
    assert all(k.startswith('proposal_gate.') for k in miss.missing_keys) and not miss.unexpected_keys
    opt=torch.optim.AdamW(model.parameters(),lr=.00002,weight_decay=.04);history=[];started=time.time();best=float('inf')
    save(folder/'config.json',dict(kind=a.kind,use_rgb=not a.tracks_only,steps=a.steps,batch=32,width=192,depth=4,heads=6,seed=seed,lr=.00002,
        root_scale_m=.1,pose_scale_m=.03,prediction='20x3 camera-space XYZ meters; separate root and relative residual tokens',selection='dev_select camera_mm + 0.5*relative_mm; test never used for selection',
        conditioning='17 aligned XYZ observations + predicted error risk + 192 spatial RGB tokens per frame + timestamps/camera poses; no GT/visibility/GT handedness',
        training='Natural observations retained, no artificial pixel/keypoint removal; diffusion adds noise to target residual only'))
    def validate(step):
        pred,_=proposals(model,data,ep,dev);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];r=metrics(pred,base,data['gt'][dev],data['valid'][dev]);history.append(dict(step=step,seconds=time.time()-started,**r));save(folder/'history.json',history)
        print(json.dumps(dict(arm=arm,**history[-1])),flush=True);return r['camera_mm']+.5*r['relative_mm']
    def checkpoint(step):torch.save(dict(model=model.state_dict(),kind=a.kind,use_rgb=not a.tracks_only,width=192,depth=4,step=step,config=json.loads((folder/'config.json').read_text())),folder/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,a.steps+1):
        model.train();g=torch.Generator(device=a.device).manual_seed(seed+step);ix=train[torch.randint(len(train),(32,),device=a.device,generator=g)];torch.manual_seed(seed+step)
        b=batch(data,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,data['gt'][ix],data['valid'][ix],data['gt_uv'][ix],data['uv_valid'][ix])
        assert torch.isfinite(loss),float(loss.detach());opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        # Warm up from 10% LR, then cosine decay to 10% over the declared run.
        scale=min(1.,.1+.9*step/200)*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        for group in opt.param_groups:group['lr']=.00002*scale
        if step%100==0:print(json.dumps(dict(arm=arm,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%500==0 or step==a.steps:
            score=validate(step)
            if score<best:best=score;checkpoint(step)
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model'])
    pred,std=proposals(model,data,ep,cal)
    np.savez_compressed(folder/'calibration.npz',indices=cal.cpu().numpy(),prediction=pred.cpu().numpy(),std=std.cpu().numpy())
    save(folder/'training_done.json',dict(complete=True,selected_step=ck['step'],best_development_score=best,seconds=time.time()-started))

if __name__=='__main__':main()

