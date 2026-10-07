"""Complete supplied 2D keypoint tracks offline; no images or GT required.

Input: image_size [1408,1408], tracks [{id, frames:[{timestamp_s, xy_px,
observed, image?}]}]. xy_px has 20 [x,y] entries or nulls. observed means
coordinates accepted by the caller, NOT automatically proven visual visibility.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch

def complete(track,model,calibration,device):
    frames=track['frames'];times=np.array([f['timestamp_s'] for f in frames],float)
    assert len(times)>0 and np.all(np.diff(times)>0),'Timestamps must increase strictly'
    xy=np.zeros((len(frames),20,2),np.float32);valid=np.zeros((len(frames),20),bool)
    for i,f in enumerate(frames):
        assert len(f['xy_px'])==20 and len(f['observed'])==20
        for j,(p,seen) in enumerate(zip(f['xy_px'],f['observed'])):
            if seen:
                assert p is not None and np.isfinite(p).all(),(i,j)
                xy[i,j]=np.asarray(p)/1408;valid[i,j]=True
    windows=[];masks=[];dts=[]
    for now in times:
        wx=np.zeros((17,20,2),np.float32);wm=np.zeros((17,20),bool);dt=np.arange(-8,9,dtype=np.float32)/6
        for k,target in enumerate(now+np.arange(-8,9)/6):
            idx=int(np.abs(times-target).argmin())
            if abs(times[idx]-target)>1/12:continue
            wx[k]=xy[idx];wm[k]=valid[idx];dt[k]=times[idx]-now
        windows.append(wx);masks.append(wm);dts.append(dt)
    from offline_kp_model import condition
    results=[]
    for start in range(0,len(frames),128):
        b=condition(torch.tensor(np.stack(windows[start:start+128]),device=device),
            torch.tensor(np.stack(masks[start:start+128]),device=device),torch.tensor(np.stack(dts[start:start+128]),device=device))
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):out=model.predict(b,samples=4)
        p=out['xy'].cpu().numpy()*1408;draws=out['samples'].cpu().numpy()*1408;std=out['std'].norm(dim=-1).cpu().numpy()*1408
        context=out['has_context'].cpu().numpy();missing=b['missing'].cpu().numpy()
        for k in range(len(p)):
            i=start+k;points=[]
            for j in range(20):
                inferred=bool(missing[k,j]);available=bool(context[k,j])
                value=p[k,j].tolist() if available else None
                if not inferred:value=frames[i]['xy_px'][j]
                radius=float(calibration['radius_factor']*(std[k,j]+calibration['std_floor_px'])) if calibration and inferred and available else None
                points.append(dict(xy_px=value,source='inferred' if inferred else 'input',available=available,
                    candidate_xy_px=draws[:,k,j].tolist() if inferred and available else [],
                    empirical_radius90_px=radius,review_required=True,reviewed=False))
            results.append(dict(timestamp_s=frames[i]['timestamp_s'],image=frames[i].get('image'),points=points))
    return dict(id=track.get('id'),frames=results)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--run',default='/mnt/why/HOT3D/experiments/offline_keypoint_diffusion_v1');ap.add_argument('--kind',choices=['dit','regression'],default='dit');ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    torch.set_num_threads(4);run=Path(a.run).resolve();sys.path.insert(0,str(run/'sealed'))
    from offline_kp_model import KeypointCompleter
    obj=json.loads(Path(a.input).read_text());assert obj['image_size']==[1408,1408],'This calibrated model is validated on HOT3D 1408x1408 coordinates'
    ck=torch.load(run/'sealed'/f'{a.kind}.pt',map_location=a.device,weights_only=False)
    model=KeypointCompleter(a.kind).to(a.device).eval();model.load_state_dict(ck['model'])
    calibration=json.loads((run/'sealed/uncertainty.json').read_text()) if a.kind=='dit' else None
    output=dict(image_size=obj['image_size'],method=a.kind,mode='offline_bidirectional',
        coordinates='canonical20 original image pixels; wrist index 5',
        note='Candidate annotations requiring review. Missing coordinates are never inputs. Observed means supplied/accepted, not proven visibility. No natural-occlusion guarantee.',
        tracks=[complete(t,model,calibration,a.device) for t in obj['tracks']])
    dest=Path(a.output);dest.parent.mkdir(exist_ok=True,parents=True);dest.write_text(json.dumps(output,indent=2))
    print(json.dumps(dict(output=str(dest),tracks=len(output['tracks']),frames=sum(len(t['frames']) for t in output['tracks']))))

if __name__=='__main__':main()
