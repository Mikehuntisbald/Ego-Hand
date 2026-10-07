import argparse,json,time
import wilor_eval_common
import numpy as np,torch
from offline_rgb_data import RUN,save
from offline_rgb_adapter import RGBAdapterCompleter
from offline_rgb_model import RGBKeypointCompleter
from train_offline_rgb import load,batch,validate

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',required=True);ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(202610036)
    folder=RUN/f'adapter_{a.kind}';folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    rows,data=load(a.device);train=torch.tensor([i for i,r in enumerate(rows) if r['role']=='train'],device=a.device)
    conf=json.loads((RUN/f'tracks_{a.kind}/config.json').read_text());dev=torch.tensor(conf['development_indices'],device=a.device)
    ck=torch.load(RUN/f'tracks_{a.kind}/best.pt',map_location=a.device,weights_only=False)
    model=RGBAdapterCompleter(a.kind).to(a.device);model.load_state_dict(ck['model'],strict=False)
    baseline=RGBKeypointCompleter(a.kind,False).to(a.device).eval();baseline.load_state_dict(ck['model'])
    b,_=batch(data,dev[:4],torch.full((4,),6,device=a.device),1+dev[:4]%4);model.eval()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict(b,samples=4)['xy'];q=baseline.predict(b,samples=4)['xy']
    assert torch.equal(p,q),'Zero-gain adapter changed the frozen temporal predictor';del baseline
    for name,p in model.named_parameters():p.requires_grad_(name.startswith(('visual','local_visual','local_gain')))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0002,weight_decay=.03)
    initial=validate(model,data,dev);best=initial['hidden_px'];history=[dict(step=0,**initial)]
    torch.save(dict(model=model.state_dict(),kind=a.kind,use_rgb=True,architecture='adapter',step=0,selection=initial),folder/'best.pt')
    save(folder/'config.json',dict(zero_gain_identity_passed=True,steps=5000,development_indices=dev.tolist(),frozen_temporal_source=f'tracks_{a.kind}/best.pt',selection='same partial-mask development subset; zero-gain candidate retained'))
    started=time.time()
    for step in range(1,5001):
        model.train();ix=train[torch.randint(len(train),(96,),device=a.device)]
        lengths=torch.tensor([1,3,6,9],device=a.device)[torch.randint(4,(96,),device=a.device)];variants=torch.randint(1,6,(96,),device=a.device)
        b,_=batch(data,ix,lengths,variants);opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,data['gt'][ix],data['valid'][ix])+.03*(model.visual_gain.square()+model.local_gain.square())
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        for g in opt.param_groups:g['lr']=.0002*(.1+.9*.5*(1+np.cos(np.pi*step/5000)))
        if step%100==0:print(json.dumps(dict(step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%500==0:
            metrics=validate(model,data,dev);entry=dict(step=step,**metrics,visual_gain=float(model.visual_gain.detach()),local_gain=float(model.local_gain.detach()));history.append(entry)
            if metrics['hidden_px']<best:
                best=metrics['hidden_px'];torch.save(dict(model=model.state_dict(),kind=a.kind,use_rgb=True,architecture='adapter',step=step,selection=metrics),folder/'best.pt')
            save(folder/'history.json',history);print(json.dumps(entry),flush=True)
    save(folder/'done.json',dict(complete=True,best_development_px=best,test_evaluated=False))

if __name__=='__main__':main()
