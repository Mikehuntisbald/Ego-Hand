import argparse,json,time
import numpy as np,torch
from hand3d_v8_common import RUN,V7,load,batch,save,initialize,predict,metrics,score,camera_bank

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['dit','regression'],default='dit');p.add_argument('--mode',choices=['rollout','one_step'],default='rollout');p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=2400);p.add_argument('--batch',type=int,default=16);p.add_argument('--lr',type=float,default=2e-5);p.add_argument('--validate-every',type=int,default=200);a=p.parse_args()
    torch.set_num_threads(4);seed=202610081;torch.manual_seed(seed);np.random.seed(seed)
    data=load(a.device);params=camera_bank().to(a.device);roles=np.array(data['roles'])
    risks=torch.load(V7/'risk_probabilities.pt',weights_only=False);prob=risks['train_oof'].to(a.device);ep=risks['joint'].to(a.device)
    train=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=a.device)
    arm=a.kind+'_'+a.mode;folder=RUN/arm;folder.mkdir(exist_ok=True)
    assert not (folder/'training_done.json').exists(), 'Preserve completed trials; choose a new run folder.'
    model,initial=initialize(a.kind,a.device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.04)
    config=dict(**vars(a),seed=seed,initial_v7_step=initial,output='20x3 camera XYZ meters',sampler='10 DDIM steps, four draws; mean then one whole-hand residual coefficient',split='Existing roles frozen; dev_select checkpoint; dev_calibrate policy; old test diagnostic only; fresh manifest required',selection='Harm<=1% in both errors, camera no worse by >0.1mm, relative >=5% better; feasible first then penalized score',rgb='frozen WiLoR32 blocks and 2D spatial stem; full 192 spatial tokens retained',training_targets='GT only in losses and training gate targets; no artificial visibility/pixel masking')
    save(folder/'config.json',config);history=[];started=time.time();best=None
    def validate(step):
        pred=predict(model,data,ep,dev);base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];gt=data['gt'][dev];valid=data['valid'][dev]
        final=metrics(pred['xyz_camera_m'],base,gt,valid);raw=metrics(pred['raw_xyz_camera_m'],base,gt,valid);key,feasible=score(final)
        row=dict(step=step,seconds=time.time()-started,feasible=feasible,trust_mean=float(pred['trust'].mean()),**final,raw=raw);history.append(row);save(folder/'history.json',history);print(json.dumps(dict(arm=arm,**row)),flush=True)
        torch.save(dict(model=model.state_dict(),kind=a.kind,step=step,config=config,metrics=row),folder/f'step_{step:05d}.pt')
        return key
    def bestsave(step):torch.save(dict(model=model.state_dict(),kind=a.kind,step=step,config=config),folder/'best.pt')
    best=validate(0);bestsave(0)
    for step in range(1,a.steps+1):
        model.train();gen=torch.Generator(device=a.device).manual_seed(seed+step);ix=train[torch.randint(len(train),(a.batch,),device=a.device,generator=gen)]
        torch.manual_seed(seed+step);b=batch(data,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.loss(b,data['gt'][ix],data['valid'][ix],data['gt_uv'][ix],data['uv_valid'][ix],params[data['feature_ids'][ix,8]],seed+step,a.mode)
        assert torch.isfinite(loss),parts;opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        scale=min(1.,.1+.9*step/100)*(.1+.9*.5*(1+np.cos(np.pi*step/a.steps)))
        for group in opt.param_groups:group['lr']=a.lr*scale
        if step%50==0:print(json.dumps(dict(arm=arm,step=step,loss=float(loss),seconds=time.time()-started,**parts)),flush=True)
        if step%a.validate_every==0 or step==a.steps:
            key=validate(step)
            if key<best:best=key;bestsave(step)
    ck=torch.load(folder/'best.pt',weights_only=False,map_location=a.device);model.load_state_dict(ck['model']);pred=predict(model,data,ep,cal)
    torch.save(dict(indices=cal,predictions=pred),folder/'calibration.pt');save(folder/'training_done.json',dict(complete=True,selected_step=ck['step'],best_score=best,seconds=time.time()-started))

if __name__=='__main__':main()
