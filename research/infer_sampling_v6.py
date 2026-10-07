"""Offline nonuniform sampling; original uniform risk context stays frozen."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from temporal_sampling_v6 import RUN,V4,OFFSETS
from infer_natural_reliability import encode_track,windows,load_models
from natural_corrector import NaturalCorrector
from natural_reliability import risk_features
from natural_policy import apply_policy
from offline_kp_model import condition
import spatial_rgb_common as s

def sampled_windows(z,offsets,device):
    times=z['times'];frames=[];chosen=[]
    assert len(offsets)==17 and offsets[8]==0 and np.all(np.diff(offsets)>0)
    for i,now in enumerate(times):
        ids=np.zeros(17,int);valid=np.zeros(17,bool);dt=np.array(offsets,np.float32)/6;used={i};ids[8]=i;valid[8]=True
        for k in sorted((k for k in range(17) if k!=8),key=lambda k:abs(offsets[k])):
            target=now+offsets[k]/6;j=int(np.abs(times-target).argmin())
            if abs(times[j]-target)>1/12 or j in used:continue
            ids[k]=j;valid[k]=True;dt[k]=times[j]-now;used.add(j)
        chosen.append([int(j) if ok else None for j,ok in zip(ids,valid)])
        frames.append(dict(xy=z['xy'][ids]*valid[:,None,None],available=z['available'][ids]&valid[:,None],dt=dt,roi=z['roi'][ids]*valid[:,None],positions=z['positions'][ids]*valid[:,None,None],rgb_valid=valid,confirmed=z['confirmed'][i],rgb=z['spatial'][ids].numpy()*valid[:,None,None]))
    return {k:torch.from_numpy(np.stack([f[k] for f in frames])).to(device) for k in frames[0]},chosen

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--kind',choices=['regression','dit'],default='regression');p.add_argument('--pattern',choices=list(OFFSETS),default='multiscale');p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    source=json.loads(Path(a.input).read_text());assert source['image_size']==[1408,1408]
    old,risk,temp,probe,policy,projection=load_models(V4,a.device);del old
    ck=torch.load(RUN/f'{a.kind}_{a.pattern}/best.pt',map_location=a.device,weights_only=False)
    model=NaturalCorrector(a.kind).to(a.device).eval();model.load_state_dict(ck['model'])
    full,_=s.common.load_model(a.device);encoder=full.backbone;del full;tracks=[];lockcount=0
    for track in source['tracks']:
        z=encode_track(track,encoder,probe,projection,a.device);uniform=windows(z,a.device);sampled,indices=sampled_windows(z,OFFSETS[a.pattern],a.device)
        # Verify reference implementation parity of the unchanged original sampler.
        original,_=sampled_windows(z,OFFSETS['uniform'],a.device)
        for key in ['xy','available','dt','roi','positions','rgb_valid','confirmed','rgb']:assert torch.allclose(original[key].float(),uniform[key].float(),atol=1e-6),key
        outputs=[]
        for start in range(0,len(track['frames']),48):
            u={k:v[start:start+48] for k,v in uniform.items()};v={k:x[start:start+48] for k,x in sampled.items()}
            features=risk_features(u['xy'],u['available'],u['dt'],u['roi'],u['scores'],u['risk_rgb'],u['positions']);prob=(risk(features)/temp).sigmoid()
            b=condition(v['xy'],v['available'],v['dt']);b.update(rgb=v['rgb'],positions=v['positions'],roi=v['roi'],rgb_valid=v['rgb_valid'],p_bad=prob,confirmed=v['confirmed'],missing=~v['confirmed'])
            with torch.autocast('cuda',dtype=torch.bfloat16):candidate=model.predict(b)['xy'].float()
            corrected,selected=apply_policy(b['linear'],{'regression':candidate},prob,v['available'][:,8],v['confirmed'],v['roi'][:,8],policy)
            outputs.append(dict(xy=corrected.cpu(),selected=selected.cpu(),p_bad=prob.cpu()))
        out={k:torch.cat([r[k] for r in outputs]) for k in outputs[0]};frames=[]
        for i,f in enumerate(track['frames']):
            pts=[]
            for j in range(20):
                locked=bool(z['confirmed'][i,j]);exists=bool(z['available'][i,j]);changed=bool(out['selected'][i,j]);xy=(out['xy'][i,j]*1408).tolist()
                if locked or (exists and not changed):xy=f['xy_px'][j]
                assert np.isfinite(xy).all()
                if locked:assert xy==f['xy_px'][j];lockcount+=1
                pts.append(dict(xy_px=xy,confirmed=locked,input_available=exists,correction_applied=changed,input_error_risk=float(out['p_bad'][i,j]) if exists else None,review_required=not locked))
            frames.append(dict(image=f['image'],timestamp_s=f['timestamp_s'],context_frame_indices=indices[i],points=pts))
        tracks.append(dict(id=track.get('id'),frames=frames))
    result=dict(kind=a.kind,pattern=a.pattern,offsets_s=(np.array(OFFSETS[a.pattern])/6).tolist(),selected_step=ck['step'],confirmed_checked=lockcount,uniform_sampler_parity_passed=True,tracks=tracks,note='Existing per-track observations only. Missing boundary/track frames stay missing; no duplicate fill. Risk uses original uniform context. Automatic coordinates require review.')
    dest=Path(a.output);dest.parent.mkdir(exist_ok=True,parents=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(dict(frames=sum(len(t['frames']) for t in tracks),confirmed_checked=lockcount,output=str(dest))))

if __name__=='__main__':main()
