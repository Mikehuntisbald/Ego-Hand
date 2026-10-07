"""Natural RGB correction; available predictions are distinct from confirmed points."""
import argparse,json,sys,hashlib
from pathlib import Path
import numpy as np,torch

@torch.inference_mode()
def encode_track(track,encoder,probe,projection,device,return_native=False):
    import spatial_rgb_common as s
    import wilor_eval_common as common
    from offline_rgb_encoder import crop_roi
    frames=track['frames'];times=np.array([f['timestamp_s'] for f in frames]);assert len(times) and np.all(np.diff(times)>0)
    xy=np.zeros((len(frames),20,2),np.float32);available=np.zeros((len(frames),20),bool);confirmed=np.zeros_like(available)
    crops=[];positions=[];rois=[];scores=[]
    for i,f in enumerate(frames):
        assert not f.get('occluder_crop_xyxy'),'Natural inference does not manufacture occlusions'
        flags=f.get('available',f.get('observed'));assert flags is not None and len(flags)==20
        locks=f.get('confirmed',[False]*20);assert len(locks)==20 and len(f['xy_px'])==20
        assert 'box_confidence' in f,'Provide the YOLO hand-box confidence; it is not a joint confidence'
        score=float(f['box_confidence']);assert 0<=score<=1;scores.append(score)
        roi=crop_roi(f['box_xyxy'])/1408;rois.append(roi)
        record=dict(image=f['image'],camera=f['camera'],clip=f.get('clip',0))
        im,pos,*_=s.prepare(record,roi*1408,[[0,0,0,0]]);crops.append(im[0]);positions.append(pos)
        for j,(p,exists,lock) in enumerate(zip(f['xy_px'],flags,locks)):
            assert not lock or exists,'A confirmed point must have a coordinate'
            if exists:
                assert p is not None and np.isfinite(p).all();xy[i,j]=np.array(p)/1408;available[i,j]=True
            confirmed[i,j]=lock
    raw=[]
    for start in range(0,len(crops),16):
        part=crops[start:start+16];n=len(part);part=part+[part[-1]]*(16-n)
        inp=common.input_tensor(part,[1]*16,rotation=1).to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=encoder(inp[:,:,:,32:-32])[-1]
        raw.append(out[:n].flatten(2).transpose(1,2).half().cpu())
    raw=torch.cat(raw);risk_rgb=(raw.float()@projection.cpu().float()).half();spatial=[]
    for start in range(0,len(raw),96):
        part=raw[start:start+96].to(device);n=len(part)
        if n<96:part=torch.cat([part,part[-1:].expand(96-n,-1,-1)])
        with torch.autocast('cuda',dtype=torch.bfloat16):out=probe.features(part)
        spatial.append(out[:n].flatten(2).transpose(1,2).half().cpu())
    result=dict(times=times,xy=xy,available=available,confirmed=confirmed,roi=np.stack(rois),scores=np.array(scores,np.float32),
        positions=np.stack(positions),risk_rgb=risk_rgb,spatial=torch.cat(spatial))
    if return_native:result['native']=raw
    return result

def windows(z,device):
    frames=[];times=z['times']
    for i,now in enumerate(times):
        ids=np.zeros(17,int);valid=np.zeros(17,bool);dt=np.arange(-8,9,dtype=np.float32)/6
        for t,target in enumerate(now+np.arange(-8,9)/6):
            j=int(np.abs(times-target).argmin())
            if abs(times[j]-target)>1/12:continue
            ids[t]=j;valid[t]=True;dt[t]=times[j]-now
        frames.append(dict(xy=z['xy'][ids]*valid[:,None,None],available=z['available'][ids]&valid[:,None],dt=dt,
            roi=z['roi'][ids]*valid[:,None],scores=z['scores'][ids]*valid,positions=z['positions'][ids]*valid[:,None,None],
            rgb_valid=valid,confirmed=z['confirmed'][i],rgb=z['spatial'][ids].numpy()*valid[:,None,None],risk_rgb=z['risk_rgb'][ids].numpy()*valid[:,None,None]))
    return {k:torch.from_numpy(np.stack([f[k] for f in frames])).to(device) for k in frames[0]}

@torch.inference_mode()
def predict_windows(w,models,risk,temperature,probe,policy):
    from natural_reliability import risk_features
    from natural_policy import apply_policy
    from offline_kp_model import condition
    output=[]
    for start in range(0,len(w['xy']),48):
        v={k:x[start:start+48] for k,x in w.items()}
        features=risk_features(v['xy'],v['available'],v['dt'],v['roi'],v['scores'],v['risk_rgb'],v['positions'])
        p_bad=(risk(features)/temperature).sigmoid()
        b=condition(v['xy'],v['available'],v['dt']);b.update(rgb=v['rgb'],positions=v['positions'],roi=v['roi'],rgb_valid=v['rgb_valid'],
            p_bad=p_bad,confirmed=v['confirmed'],missing=~v['confirmed'])
        candidates={}
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for arm,model in models.items():candidates[arm]=model.predict(b)['xy'].float()
            x=v['rgb'][:,8].float().transpose(1,2).reshape(-1,128,16,12)
            candidates['rgb_probe']=probe.decode(x,v['positions'][:,8],v['roi'][:,8])['xy'].float()
        candidate_risk={}
        for arm in candidates:
            test_xy=v['xy'].clone();test_available=v['available'].clone();test_xy[:,8]=candidates[arm];test_available[:,8]=True
            feat=risk_features(test_xy,test_available,v['dt'],v['roi'],v['scores'],v['risk_rgb'],v['positions'])
            candidate_risk[arm]=(risk(feat)/temperature).sigmoid()
        args=(b['linear'],candidates,p_bad,v['available'][:,8],v['confirmed'],v['roi'][:,8],policy)
        result,selected=apply_policy(*args,candidate_risk=candidate_risk) if policy.get('risk_margin') is not None else apply_policy(*args)
        output.append(dict(xy=result.cpu(),selected=selected.cpu(),p_bad=p_bad.cpu(),**{k:x.cpu() for k,x in candidates.items()},
            **{k+'_risk':x.cpu() for k,x in candidate_risk.items()}))
    return {k:torch.cat([o[k] for o in output]) for k in output[0]}

def load_models(run,device):
    from natural_corrector import NaturalCorrector
    from natural_reliability import RiskHead
    from spatial_rgb_model import SpatialHead
    manifest=json.loads((run/'sealed/manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((run/'sealed'/name).read_bytes()).hexdigest()==digest,name
    models={}
    for arm in ['dit','regression']:
        ck=torch.load(run/'sealed'/f'{arm}.pt',weights_only=False,map_location=device);model=NaturalCorrector(arm).to(device).eval();model.load_state_dict(ck['model']);models[arm]=model
    ck=torch.load(run/'sealed/risk_joint.pt',weights_only=False,map_location=device);risk=RiskHead(ck['dim']).to(device).eval();risk.load_state_dict(ck['model'])
    temperature=json.loads((run/'sealed/risk_calibration.json').read_text())['joint']['temperature']
    probe=SpatialHead().to(device).eval();probe.load_state_dict(torch.load(run/'sealed/rgb_probe.pt',weights_only=False,map_location=device)['model'])
    policy=json.loads((run/'sealed/policy.json').read_text())['policy']
    projection=torch.load(run/'sealed/risk_projection.pt',weights_only=False)
    return models,risk,temperature,probe,policy,projection

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--run',default='/mnt/why/HOT3D/experiments/natural_reliability_v4');p.add_argument('--device',default='cuda:0');a=p.parse_args()
    torch.set_num_threads(4);run=Path(a.run).resolve();dest=Path(a.output).resolve();obj=json.loads(Path(a.input).resolve().read_text());assert obj['image_size']==[1408,1408]
    sys.path.insert(0,str(run/'sealed'));models,risk,temp,probe,policy,projection=load_models(run,a.device)
    import wilor_eval_common as common
    full,_=common.load_model(a.device);encoder=full.backbone;del full;tracks=[]
    for track in obj['tracks']:
        z=encode_track(track,encoder,probe,projection,a.device);w=windows(z,a.device);out=predict_windows(w,models,risk,temp,probe,policy);frames=[]
        for i,f in enumerate(track['frames']):
            points=[]
            for j in range(20):
                locked=bool(z['confirmed'][i,j]);changed=bool(out['selected'][i,j]);exists=bool(z['available'][i,j])
                xy=(out['xy'][i,j]*1408).tolist()
                if locked or (exists and not changed):xy=f['xy_px'][j]
                input_risk=float(out['p_bad'][i,j]);flagged=exists and input_risk>=policy.get('threshold',1.1)
                reason='confirmed' if locked else 'missing_input' if not exists else 'selected_correction' if changed else 'retained_unconfirmed'
                if flagged and not changed and not locked:
                    distance=float((out[policy['arm']][i,j]-torch.tensor(z['xy'][i,j])).norm())/max(float(np.mean(z['roi'][i,2:]-z['roi'][i,:2])),.01)
                    reason='correction_exceeds_validated_range' if distance>policy.get('max_delta_roi',.5) else 'candidate_not_more_reliable'
                points.append(dict(xy_px=xy,source='confirmed_input' if locked else ('corrected_prediction' if changed and exists else 'completed_missing' if not exists else 'unconfirmed_input'),
                    confirmed=locked,input_available=exists,input_error_risk=input_risk if exists else None,
                    correction_applied=changed,review_required=not locked,
                    decision_reason=reason,needs_special_review=bool(flagged and not changed and not locked),
                    candidate_error_scores={k:float(out[k+'_risk'][i,j]) for k in ['dit','regression','rgb_probe']},
                    candidates={k:(out[k][i,j]*1408).tolist() for k in ['dit','regression','rgb_probe']}))
            frames.append(dict(image=f['image'],timestamp_s=f['timestamp_s'],points=points))
        tracks.append(dict(id=track.get('id'),frames=frames))
    result=dict(image_size=[1408,1408],mode='natural_rgb_selective_correction',policy=policy,tracks=tracks,
        note='Input error risk is calibrated for original WiLoR predictions, not visibility or correctness of the corrected output. Automatic points still require review.')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=str(dest),frames=sum(len(t['frames']) for t in tracks))))

if __name__=='__main__':main()
