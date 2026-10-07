"""Fix local preservation and calibrate only on P0003, without test result access."""
import argparse,json,time
import numpy as np,torch
from residual_models import ResidualModel
from temporal_residual import TemporalResidualModel,attach_context
from train_coarse_pose import RUN
from metrics_3d import compare,EVAL_INDICES

def proposals(model,data,ids,device,seed):
    out=[];gen=torch.Generator(device=device).manual_seed(seed)
    with torch.inference_mode():
        for start in range(0,len(ids),256):
            ix=ids[start:start+256];c=data['coarse'][ix].to(device);f=data['confidence'][ix].to(device);rgb=data['rgb'][ix].to(device).float()
            attach_context(model,data,ix,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model.propose(c,f,rgb,steps=10,samples=4 if model.kind=='dit' else 1,generator=gen)
            out.append(p.float().cpu())
    return torch.cat(out)

def calibrate(model,data,ids,prop,device):
    gates=[]
    with torch.inference_mode():
        for start in range(0,len(ids),256):
            ix=ids[start:start+256]
            attach_context(model,data,ix,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):g=model.gates(prop[start:start+256].to(device),data['coarse'][ix].to(device),data['confidence'][ix].to(device),data['rgb'][ix].to(device).float())[1]
            gates.append(g.float().cpu())
    gates=torch.cat(gates);coarse=data['coarse'][ids];gt=data['gt'][ids]
    low=data['confidence'][ids,1:].numpy()<float(torch.quantile(data['confidence'][ids,1:][:,EVAL_INDICES].reshape(-1),.2))
    low[:,5]=False
    before_relative=((coarse-coarse[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1).numpy()*1000
    precise=before_relative[:,EVAL_INDICES]<=10
    base=compare(coarse.numpy(),coarse.numpy(),gt.numpy())
    best=None;candidates=[]
    for threshold in [0,.25,.5,.75,.9,.95,.99]:
        selected=gates*(gates>=threshold)
        for strength in [0,.05,.1,.25,.5,1.]:
            pred=model.apply_gates(prop,coarse,selected*strength).numpy()
            result=compare(coarse.numpy(),pred,gt.numpy())
            er=np.linalg.norm((pred-pred[:,5:6])-(gt.numpy()-gt.numpy()[:,5:6]),axis=-1)*1000
            relharm=float((((er[:,EVAL_INDICES]>before_relative[:,EVAL_INDICES]+1)&precise).sum())/max(1,precise.sum()))
            low_error=float(er[low].mean())
            admissible=result['correct_joints_harmed_fraction']<=.05 and relharm<=.05 and result['refined']['mpjpe19_mm']<=base['coarse']['mpjpe19_mm']+.2
            score=result['refined']['mpjpe19_mm']+result['refined']['wrist_relative_mpjpe19_mm']+.5*low_error
            result['refined'].pop('sample_mpjpe19_mm');result['coarse'].pop('sample_mpjpe19_mm')
            row=dict(threshold=threshold,strength=strength,admissible=bool(admissible),score=score,tune=result,
                     relative_correct_harmed=relharm,low_confidence_relative_mm=low_error)
            candidates.append(row)
            if admissible and (best is None or score<best['score']):best=row
    assert best is not None
    return best,candidates

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['dit','regression'],default='dit');ap.add_argument('--device',default='cuda:2');ap.add_argument('--temporal',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(20261003);device=a.device
    data=torch.load(RUN/'coarse_cache.pt',map_location='cpu',weights_only=False)
    if a.temporal:data['temporal']=torch.load(RUN/'temporal_context.pt',map_location='cpu',weights_only=False)
    train=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='residual']);tune=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='tune'])
    model=(TemporalResidualModel(kind=a.kind) if a.temporal else ResidualModel(kind=a.kind)).to(device)
    source=RUN/(f'temporal_residual_{a.kind}' if a.temporal else f'residual_{a.kind}')
    model.load_state_dict(torch.load(source/'denoiser_best.pt',map_location=device,weights_only=False)['model'])
    for p in model.parameters():p.requires_grad_(False)
    model.eval();bank=[proposals(model,data,train,device,s) for s in [1901,1902]];tune_prop=proposals(model,data,tune,device,81003)
    for p in model.gate.parameters():p.requires_grad_(True)
    opt=torch.optim.AdamW(model.gate.parameters(),lr=.0003,weight_decay=.001);folder=RUN/(f'temporal_selective_{a.kind}' if a.temporal else f'selective_{a.kind}');folder.mkdir(exist_ok=True)
    # Locality invariant: a closed point gate retains that point even with an open
    # root gate and a nonzero root proposal.
    test_prop=torch.zeros(2,21,3,device=device);test_prop[:,0,0]=1
    test_g=torch.zeros(2,21,device=device);test_g[:,0]=1
    c=data['coarse'][:2].to(device);p=model.apply_gates(test_prop,c,test_g)
    assert torch.equal(p[:,0],c[:,0]) and not torch.equal(p[:,5],c[:,5])
    started=time.time();best_score=float('inf');history=[]
    for step in range(1,2001):
        positions=torch.randint(len(train),(256,));ix=train[positions];prop=bank[step%2][positions].to(device)
        attach_context(model,data,ix,device)
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.selective_gate_loss(prop,data['gt'][ix].to(device),data['coarse'][ix].to(device),data['confidence'][ix].to(device),data['rgb'][ix].to(device).float())
        assert torch.isfinite(loss),parts;loss.backward();torch.nn.utils.clip_grad_norm_(model.gate.parameters(),1.);opt.step()
        if step%250==0:
            chosen,grid=calibrate(model,data,tune,tune_prop,device)
            row=dict(step=step,train=parts,calibration=chosen,seconds=time.time()-started);history.append(row)
            checkpoint=dict(model=model.state_dict(),kind=a.kind,stage='selective_gate',step=step,calibration=chosen,
                            history=history,gate_application='local camera-space XYZ including root correction')
            torch.save(checkpoint,folder/'last.pt')
            if chosen['score']<best_score:
                best_score=chosen['score'];torch.save(checkpoint,folder/'best.pt');(folder/'calibration_grid.json').write_text(json.dumps(grid,indent=2))
            (folder/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    (folder/'done.json').write_text(json.dumps(dict(completed=True,checkpoint=str(folder/'best.pt'))))

if __name__=='__main__':main()
