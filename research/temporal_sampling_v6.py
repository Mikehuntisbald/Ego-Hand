"""Fixed-token temporal sampling trial using original prediction-only tracks."""
import argparse,json,time,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
import spatial_rgb_common as s
from natural_reliability import RUN as V4
from natural_corrector import NaturalCorrector,natural_batch
from train_spatial_temporal import load
from evaluate_natural_reliability import measure
from natural_policy import apply_policy
RUN=s.common.ROOT/'experiments/temporal_sampling_v6';RUN.mkdir(exist_ok=True)
OFFSETS={'uniform':list(range(-8,9)), 'wide':list(range(-24,25,3)), 'multiscale':[-24,-16,-10,-6,-4,-3,-2,-1,0,1,2,3,4,6,10,16,24]}

def prepare():
    records,index=s.records_and_index();rows=json.loads((s.OLD/'rows.json').read_text());w=torch.load(s.OLD/'windows.pt',weights_only=False)
    times={}
    for source in sorted({r['source'] for r in rows}):
        frames=json.loads((s.common.ROOT/'experiments'/source/'locked_frames.json').read_text())
        for f in frames:times[(source,f['sequence'],f['clip'],f['frame'])]=f['timestamp_ns']*1e-9
    # Union of existing tracked observations; no GT side, visibility or pose used.
    groups=defaultdict(dict);observation={}
    def group(r):return (r['source'],r['sequence'],r['clip'],r['track_id'])
    owners={}
    for j,r in enumerate(rows):
        key=group(r)
        for k,fid in enumerate(index['feature_ids'][j].tolist()):
            if not fid:continue
            rec=records[fid-1];frame=rec['frame'];assert rec['sequence']==r['sequence'] and rec['clip']==r['clip']
            assert fid not in owners or owners[fid]==key;owners[fid]=key
            assert frame not in groups[key] or groups[key][frame]==fid
            groups[key][frame]=fid
            if fid in observation:
                assert torch.equal(observation[fid][0],w['xy'][j,k]) and torch.equal(observation[fid][1],w['observed'][j,k])
            else:observation[fid]=(w['xy'][j,k],w['observed'][j,k])
    report={};audit=json.loads((V4/'delivery/hardcase_review_queue.json').read_text());failed=[q for q in audit if q['after_px']>40]
    for name,offsets in OFFSETS.items():
        xy=torch.zeros_like(w['xy']);obs=torch.zeros_like(w['observed']);dt=torch.tensor(offsets,dtype=torch.float32)[None].repeat(len(rows),1)/6;fids=torch.zeros_like(index['feature_ids'])
        for j,r in enumerate(rows):
            now=times[(r['source'],r['sequence'],r['clip'],r['frame'])]
            for k,off in enumerate(offsets):
                frame=r['frame']+5*off;fid=groups[group(r)].get(frame,0)
                if not fid:continue
                fids[j,k]=fid;xy[j,k],obs[j,k]=observation[fid];dt[j,k]=times[(r['source'],r['sequence'],r['clip'],frame)]-now
            assert fids[j,8]==index['feature_ids'][j,8]
            active=fids[j][fids[j]>0];assert len(active)==len(torch.unique(active))
            selected_dt=dt[j][fids[j]>0];assert torch.all(selected_dt[1:]>selected_dt[:-1])
        assert torch.equal(xy[:,8],w['xy'][:,8]) and torch.equal(obs[:,8],w['observed'][:,8])
        if name=='uniform':
            assert torch.equal(fids,index['feature_ids']) and torch.equal(xy,w['xy']) and torch.equal(obs,w['observed'])
            assert torch.allclose(dt,w['dt'],atol=1e-6),float((dt-w['dt']).abs().max())
        torch.save(dict(xy=xy,observed=obs,dt=dt,feature_ids=fids),RUN/f'{name}_windows.pt')
        cases=[]
        for q in failed:
            j=q['window_index'];cases.append(dict(id=q['id'],target_frame=q['frame'],sampled_frames=[records[fid-1]['frame'] if fid else None for fid in fids[j].tolist()],actual_dt_s=[float(t) if fid else None for fid,t in zip(fids[j].tolist(),dt[j].tolist())]))
        report[name]=dict(offsets_s=(np.array(offsets)/6).tolist(),valid_frame_mean=float((fids>0).float().sum(1).mean()),valid_frame_median=float((fids>0).float().sum(1).median()),fully_populated_windows=int((fids>0).all(1).sum()),cases=cases)
    s.save(RUN/'sampling_checks.json',dict(passed=True,old_uniform_exact=True,center_unchanged=True,no_duplicate_padding=True,no_cross_track_or_clip=True,patterns=report))
    s.save(RUN/'protocol.json',dict(schedules={k:(np.array(v)/6).tolist() for k,v in OFFSETS.items()},steps=1500,batch=48,seed=202610061,lr=2e-5,visual='unchanged frozen v3 spatial features',initialization='v4 sealed head for each kind',selection='dev_select raw mean error, includes step0',gate='same frozen v4 regression gate and original-window risk scores for every arm',track='same prediction-only track associations; no reconnect and no GT selection',density='nearest interval remains 1/6s; no new 30Hz observations',scope='existing inspected test set, development diagnostic only; 47 hardcases not used for fitting or checkpoint selection',code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    print(json.dumps(report,indent=2))

@torch.no_grad()
def predict(model,data,prob,ids):
    model.eval();parts=[]
    for start in range(0,len(ids),48):
        ix=ids[start:start+48];b=natural_batch(data,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):parts.append(model.predict(b)['xy'].float())
    return torch.cat(parts)

def train(a):
    torch.manual_seed(202610061);np.random.seed(202610061)
    folder=RUN/f'{a.kind}_{a.pattern}';folder.mkdir(exist_ok=True)
    rows,data=load(a.device);data.update({k:v.to(a.device) for k,v in torch.load(RUN/f'{a.pattern}_windows.pt',weights_only=False).items()})
    roles=np.array(torch.load(V4/'features.pt',weights_only=False,mmap=True)['roles']);probs=torch.load(V4/'probabilities.pt',weights_only=False)
    prob=probs['train_oof'].to(a.device);ep=probs['joint'].to(a.device)
    ids=torch.tensor(np.where(roles=='train')[0],device=a.device);dev=torch.tensor(np.where(roles=='dev_select')[0],device=a.device)
    model=NaturalCorrector(a.kind).to(a.device);model.load_state_dict(torch.load(V4/'sealed'/f'{a.kind}.pt',weights_only=False,map_location=a.device)['model'])
    opt=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.04);history=[];started=time.time()
    def validate(step):
        p=predict(model,data,ep,dev);b=natural_batch(data,dev,ep);m=data['valid'][dev].clone()&data['observed'][dev,8];m[:,5]=False
        scores=measure(p,b['linear'],data['gt'][dev],m);history.append(dict(step=step,seconds=time.time()-started,**scores));s.save(folder/'history.json',history)
        print(json.dumps(dict(arm=folder.name,**history[-1])),flush=True);return scores['mean_px']
    def checkpoint(step):torch.save(dict(model=model.state_dict(),kind=a.kind,pattern=a.pattern,offsets=OFFSETS[a.pattern],step=step),folder/'best.pt')
    best=validate(0);checkpoint(0)
    for step in range(1,1501):
        model.train();g=torch.Generator(device=a.device).manual_seed(202610061+step);ix=ids[torch.randint(len(ids),(48,),device=a.device,generator=g)];torch.manual_seed(202610061+step)
        b=natural_batch(data,ix,prob)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,data['gt'][ix],data['valid'][ix])
        assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        for group in opt.param_groups:group['lr']=2e-5*(.1+.9*.5*(1+np.cos(np.pi*step/1500)))
        if step%100==0:print(json.dumps(dict(arm=folder.name,step=step,loss=float(loss.detach()),seconds=time.time()-started)),flush=True)
        if step%300==0:
            score=validate(step)
            if score<best:best=score;checkpoint(step)
    ck=torch.load(folder/'best.pt',map_location=a.device,weights_only=False);model.load_state_dict(ck['model'])
    test=torch.tensor(np.where(roles=='test')[0],device=a.device);pred=predict(model,data,ep,test);b=natural_batch(data,test,ep)
    base=b['linear'];available=data['observed'][test,8];mask=data['valid'][test].clone()&available;mask[:,5]=False
    policy=json.loads((V4/'sealed/policy.json').read_text())['policy'];assert policy['agreement_roi'] is None
    final,selected=apply_policy(base,{'regression':pred},ep[test],available,torch.zeros_like(available),b['roi'][:,8],policy)
    audit=json.loads((V4/'delivery/hardcase_review_queue.json').read_text());lookup={v:i for i,v in enumerate(test.tolist())};hard=torch.tensor([lookup[q['window_index']] for q in audit],device=a.device)
    report=dict(kind=a.kind,pattern=a.pattern,selected_step=ck['step'],raw=measure(pred,base,data['gt'][test],mask),gated=measure(final,base,data['gt'][test],mask,selected),hardcases=measure(final[hard],base[hard],data['gt'][test][hard],mask[hard],selected[hard]),cases=[])
    for q in audit:
        j=lookup[q['window_index']];report['cases'].append(dict(id=q['id'],v4_px=q['after_px'],error_px=float(((final[j]-data['gt'][test[j]]).norm(dim=-1)*1408)[mask[j]].mean()),valid_frames=int((data['feature_ids'][test[j]]>0).sum())))
    s.save(folder/'results.json',report)
    np.savez_compressed(folder/'predictions.npz',window_indices=test.cpu().numpy(),prediction=final.cpu().numpy(),raw=pred.cpu().numpy(),gt=data['gt'][test].cpu().numpy(),base=base.cpu().numpy(),mask=mask.cpu().numpy())
    s.save(folder/'done.json',dict(complete=True,seconds=time.time()-started))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train']);p.add_argument('--device',default='cuda:0');p.add_argument('--kind',default='regression');p.add_argument('--pattern',choices=list(OFFSETS),default='multiscale');a=p.parse_args();torch.set_num_threads(4)
    prepare() if a.stage=='prepare' else train(a)
