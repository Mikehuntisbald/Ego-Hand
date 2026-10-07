"""Subject OOF refiners, sequence-disjoint gate training and held calibration."""
import argparse,json,random,time
import numpy as np,torch
from residual_models import ResidualModel
from metrics_3d import compare,EVAL_INDICES
from oof_common import RUN,save,sha
from oof_calibration import active_data,fit_uncertainty,fit_probability,calibration_grid

def propose(model,data,ids,device,seed):
    generator=torch.Generator(device=device).manual_seed(seed);out=[];logits=[]
    model.eval()
    with torch.inference_mode():
        for start in range(0,len(ids),256):
            ix=ids[start:start+256];c=data['coarse'][ix].to(device);f=data['confidence'][ix].to(device);rgb=data['rgb'][ix].to(device).float()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p=model.propose(c,f,rgb,steps=10,samples=4 if model.kind=='dit' else 1,generator=generator)
                l=model.gates(p,c,f,rgb)[0]
            out.append(p.float().cpu());logits.append(l.float().cpu())
    return torch.cat(out),torch.cat(logits)

def summary_compare(coarse,refined,gt):
    result=compare(coarse.numpy(),refined.numpy(),gt.numpy())
    for k in ['coarse','refined']:result[k].pop('sample_mpjpe19_mm')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['oof_dit','oof_regression','in_subject_dit'],required=True);ap.add_argument('--device',required=True);a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261003);np.random.seed(20261003);random.seed(20261003)
    folder=RUN/a.method;folder.mkdir(exist_ok=True)
    if (folder/'done.json').exists():return
    protocol=json.loads((RUN/'protocol.json').read_text());raw=torch.load(RUN/'oof_cache.pt',map_location='cpu',weights_only=False)
    ids={k:torch.tensor([i for i,r in enumerate(raw['rows']) if r['role']==k]) for k in ['denoise','gate_fit','select','calibrate']}
    assert all(len(v)>100 for v in ids.values())
    seqs={k:{raw['rows'][i]['sequence'] for i in v.tolist()} for k,v in ids.items()}
    assert all(not seqs[x]&seqs[y] for x in seqs for y in seqs if x!=y)
    uncal=active_data(raw,a.method);t=ids['denoise']
    fit=fit_uncertainty(uncal['coarse'][t],raw['gt'][t],uncal['confidence'][t],[raw['rows'][i]['subject'] for i in t.tolist()])
    save(folder/'uncertainty_calibration.json',fit);data=active_data(raw,a.method,fit)
    kind='regression' if a.method=='oof_regression' else 'dit';model=ResidualModel(kind=kind).to(a.device)
    opt=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.01)
    history=json.loads((folder/'history.json').read_text()) if (folder/'history.json').exists() else []
    started=time.time();best=float('inf');finished_denoiser=False
    if (folder/'denoiser_last.pt').exists():
        finished_denoiser=torch.load(folder/'denoiser_last.pt',map_location='cpu',weights_only=False)['step']==protocol['steps']
    for step in range(protocol['steps']+1 if finished_denoiser else 1,protocol['steps']+1):
        model.train();ix=t[torch.randint(len(t),(256,))]
        c=data['coarse'][ix].to(a.device);f=data['confidence'][ix].to(a.device);rgb=data['rgb'][ix].to(a.device).float();gt=data['gt'][ix].to(a.device)
        c=c+torch.randn_like(c)*.001;opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.prediction_loss(gt,c,f,rgb)
        assert torch.isfinite(loss),parts
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step%100==0:print(json.dumps(dict(stage='denoiser',step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%500==0:
            si=ids['select'];p,_=propose(model,data,si,a.device,81003)
            result=summary_compare(data['coarse'][si],data['coarse'][si]+model.unpack(p),data['gt'][si])
            score=result['refined']['mpjpe19_mm']+result['refined']['wrist_relative_mpjpe19_mm']
            row=dict(stage='denoiser',step=step,selection=result,seconds=time.time()-started);history.append(row)
            ck=dict(model=model.state_dict(),method=a.method,step=step,uncertainty_calibration=fit)
            torch.save(ck,folder/'denoiser_last.pt')
            if score<best:best=score;torch.save(ck,folder/'denoiser_best.pt')
            save(folder/'history.json',history);print(json.dumps(row),flush=True)
    ck=torch.load(folder/'denoiser_best.pt',map_location=a.device,weights_only=False);model.load_state_dict(ck['model']);denoiser_step=ck['step']
    torch.manual_seed(20261003) # Gate stage remains reproducible after completed-denoiser resume.
    for param in model.parameters():param.requires_grad_(False)
    gi=ids['gate_fit'];bank=[propose(model,data,gi,a.device,seed)[0] for seed in [501,502]]
    for param in model.gate.parameters():param.requires_grad_(True)
    opt=torch.optim.AdamW(model.gate.parameters(),lr=.0003,weight_decay=.001);best=float('inf')
    for step in range(1,protocol['gate_steps']+1):
        pos=torch.randint(len(gi),(256,));ix=gi[pos];p=bank[step%2][pos].to(a.device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.selective_gate_loss(p,data['gt'][ix].to(a.device),data['coarse'][ix].to(a.device),data['confidence'][ix].to(a.device),data['rgb'][ix].to(a.device).float())
        assert torch.isfinite(loss),parts
        loss.backward();torch.nn.utils.clip_grad_norm_(model.gate.parameters(),1.);opt.step()
        if step%250==0:
            si=ids['select'];p,l=propose(model,data,si,a.device,81003)
            # Fixed conservative selection setting; independent calibration
            # sequence is not accessed until the gate checkpoint is frozen.
            g=l.sigmoid()*model.mask.squeeze(-1).cpu()*.25
            refined=model.apply_gates(p,data['coarse'][si],g);result=summary_compare(data['coarse'][si],refined,data['gt'][si])
            er=((refined-refined[:,5:6])-(data['gt'][si]-data['gt'][si,5:6])).norm(dim=-1)*1000
            before=((data['coarse'][si]-data['coarse'][si,5:6])-(data['gt'][si]-data['gt'][si,5:6])).norm(dim=-1)*1000
            precise=before[:,EVAL_INDICES]<=10
            rh=float(((er[:,EVAL_INDICES]>before[:,EVAL_INDICES]+1)&precise).sum()/precise.sum().clamp_min(1))
            score=result['refined']['mpjpe19_mm']+result['refined']['wrist_relative_mpjpe19_mm']+200*max(0,result['correct_joints_harmed_fraction']-.025)+200*max(0,rh-.025)
            row=dict(stage='gate',step=step,selection=result,relative_harm=rh,score=score,train=parts,seconds=time.time()-started);history.append(row)
            ck=dict(model=model.state_dict(),method=a.method,step=step,denoiser_step=denoiser_step,uncertainty_calibration=fit)
            torch.save(ck,folder/'gate_last.pt')
            if score<best:best=score;torch.save(ck,folder/'gate_best.pt')
            save(folder/'history.json',history);print(json.dumps(row),flush=True)
    ck=torch.load(folder/'gate_best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ck['model']);model.eval()
    ci=ids['calibrate'];prop,logits=propose(model,data,ci,a.device,83003)
    # Calibration math stays on CPU; model.scale/mask must match these tensors.
    model.cpu();probability=fit_probability(logits,prop,data['coarse'][ci],data['gt'][ci],model)
    operating,grid=calibration_grid(model,data['coarse'][ci],data['gt'][ci],prop,logits,probability,data['confidence'][ci])
    ck['calibration']=operating;ck['model']=model.state_dict();ck['provenance']=dict(protocol_sha256=sha(RUN/'protocol.json'),
        denoiser_sequences=sorted(seqs['denoise']),gate_fit_sequences=sorted(seqs['gate_fit']),
        selection_sequence=sorted(seqs['select']),calibration_sequence=sorted(seqs['calibrate']),rgb_basis='common frozen COCO',initialized_from_v1=False)
    torch.save(ck,folder/'best.pt');save(folder/'calibration_grid.json',grid);save(folder/'gate_calibration.json',operating)
    save(folder/'done.json',dict(complete=True,checkpoint=str(folder/'best.pt'),sha256=sha(folder/'best.pt'),calibration=operating,
        train_samples=len(t),gate_samples=len(gi),selection_samples=len(ids['select']),calibration_samples=len(ci)))
    print(json.dumps(dict(method=a.method,done=True,operating=operating)),flush=True)
if __name__=='__main__':main()
