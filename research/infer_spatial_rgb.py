"""Offline dense-RGB completion. Frames require camera, image, box and timestamps.

Keypoint order and 1408x1408 native Aria image coordinates match RGB v2. The
optional occluder_crop_xyxy is applied to pixels before the encoder and removes
all coordinate observations in that frame. Every inferred point needs review.
"""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np,torch

@torch.inference_mode()
def prepare_track(track,encoder,probe,device):
    import spatial_rgb_common as s
    import wilor_eval_common as common
    from offline_rgb_encoder import crop_roi
    frames=track['frames'];times=np.array([f['timestamp_s'] for f in frames],float)
    assert len(times) and np.all(np.diff(times)>0)
    xy=np.zeros((len(frames),20,2),np.float32);observed=np.zeros((len(frames),20),bool)
    crops=[];positions=[];rois=[]
    for i,f in enumerate(frames):
        assert len(f['xy_px'])==20 and len(f['observed'])==20
        roi=crop_roi(f['box_xyxy'])/1408;rois.append(roi)
        rect=f.get('occluder_crop_xyxy') or [0,0,0,0];pixel_mask=rect[2]>rect[0] and rect[3]>rect[1]
        if encoder is not None:
            record=dict(image=f['image'],camera=f['camera'],clip=f.get('clip',0))
            im,pos,*_=s.prepare(record,roi*1408,[rect],color=f.get('occluder_color',128))
            crops.append(im[0]);positions.append(pos)
        else:positions.append(np.zeros((192,2),np.float32))
        for j,(p,seen) in enumerate(zip(f['xy_px'],f['observed'])):
            if seen and not pixel_mask:
                assert p is not None and np.isfinite(p).all()
                xy[i,j]=np.asarray(p)/1408;observed[i,j]=True
    dense=[]
    if encoder is not None:
        for start in range(0,len(crops),16):
            part=crops[start:start+16];n=len(part);part=part+[part[-1]]*(16-n)
            x=common.input_tensor(part,[1]*16,rotation=1).to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):raw=encoder(x[:,:,:,32:-32])[-1]
            dense.append(raw[:n].flatten(2).transpose(1,2).half())
        dense=torch.cat(dense);features=[]
        for start in range(0,len(dense),96):
            part=dense[start:start+96];n=len(part)
            if n<96:part=torch.cat([part,part[-1:].expand(96-n,-1,-1)])
            with torch.autocast('cuda',dtype=torch.bfloat16):z=probe.features(part)
            features.append(z[:n].flatten(2).transpose(1,2).half().cpu())
        bank=torch.cat(features)
    else:bank=torch.zeros(len(frames),192,128,dtype=torch.float16)
    return dict(times=times,xy=xy,observed=observed,roi=np.stack(rois),positions=np.stack(positions),bank=bank)

@torch.inference_mode()
def complete(track,model,encoder,probe,calibration,device):
    from offline_kp_model import condition
    z=prepare_track(track,encoder,probe,device);frames=track['frames'];times=z['times'];windows=[]
    for now in times:
        wx=np.zeros((17,20,2),np.float32);wm=np.zeros((17,20),bool);dt=np.arange(-8,9,dtype=np.float32)/6
        ci=np.zeros(17,np.int64);cv=np.zeros(17,bool)
        for k,target in enumerate(now+np.arange(-8,9)/6):
            i=int(np.abs(times-target).argmin())
            if abs(times[i]-target)>1/12:continue
            wx[k]=z['xy'][i];wm[k]=z['observed'][i];dt[k]=times[i]-now;ci[k]=i;cv[k]=True
        windows.append((wx,wm,dt,ci,cv))
    results=[]
    for start in range(0,len(frames),48):
        wx,wm,dt,ci,cv=map(np.stack,zip(*windows[start:start+48]))
        b=condition(torch.tensor(wx,device=device),torch.tensor(wm,device=device),torch.tensor(dt,device=device))
        valid=torch.tensor(cv,device=device)
        b.update(rgb=z['bank'][ci].to(device)*valid[:,:,None,None],positions=torch.tensor(z['positions'][ci],device=device)*valid[:,:,None,None],
            roi=torch.tensor(z['roi'][ci],device=device)*valid[:,:,None],rgb_valid=valid)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model.predict(b,samples=4)
        xy=out['xy'].cpu().numpy()*1408;draws=out['samples'].cpu().numpy()*1408;std=out['std'].norm(dim=-1).cpu().numpy()*1408
        context=out['has_context'].cpu().numpy();missing=b['missing'].cpu().numpy()
        for k in range(len(xy)):
            i=start+k;points=[]
            for j in range(20):
                inferred=bool(missing[k,j]);available=bool(context[k,j] or model.use_rgb)
                value=xy[k,j].tolist() if available else None
                if not inferred:value=frames[i]['xy_px'][j]
                radius=float(calibration['factor']*(std[k,j]+calibration['floor_px'])) if calibration and inferred and available else None
                points.append(dict(xy_px=value,source='inferred' if inferred else 'input',available=available,
                    temporal_keypoint_context=bool(context[k,j]),candidate_xy_px=draws[:,k,j].tolist() if inferred and available else [],
                    empirical_radius90_px=radius,review_required=True,reviewed=False))
            results.append(dict(timestamp_s=frames[i]['timestamp_s'],image=frames[i]['image'],points=points))
    return dict(id=track.get('id'),frames=results)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--run',default='/mnt/why/HOT3D/experiments/offline_keypoint_spatial_v3');p.add_argument('--device',default='cuda:0')
    p.add_argument('--arm',choices=['rgb_dit','rgb_regression','tracks_dit','tracks_regression'],default='rgb_dit');a=p.parse_args()
    torch.set_num_threads(4);run=Path(a.run).resolve();source=Path(a.input).resolve();dest=Path(a.output).resolve();sys.path.insert(0,str(run/'sealed'))
    from spatial_temporal_model import SpatialTemporalCompleter
    from spatial_rgb_model import SpatialHead
    import wilor_eval_common as common
    obj=json.loads(source.read_text());assert obj['image_size']==[1408,1408]
    seal=json.loads((run/'sealed/selection.json').read_text());path=run/'sealed'/f'{a.arm}.pt'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(path)==seal['models'][a.arm]
    for name,h in seal['code'].items():assert digest(run/'sealed'/name)==h,name
    ck=torch.load(path,map_location=a.device,weights_only=False)
    model=SpatialTemporalCompleter(ck['kind'],ck['use_rgb']).to(a.device).eval();model.load_state_dict(ck['model'])
    encoder=probe=None
    if model.use_rgb:
        assert digest(common.RUN/'assets/wilor_final.mirror.ckpt')==seal['encoder_sha256']
        assert digest(run/'sealed/rgb_probe.pt')==seal['projection_sha256']
        full,_=common.load_model(a.device);encoder=full.backbone;del full
        assert len(encoder.blocks)==32 and not encoder.skip_blocks
        probe=SpatialHead().to(a.device).eval();probe.load_state_dict(torch.load(run/'sealed/rgb_probe.pt',map_location=a.device,weights_only=False)['model'])
    calibration=json.loads((run/'sealed/uncertainty.json').read_text()) if a.arm=='rgb_dit' else None
    result=dict(image_size=obj['image_size'],arm=a.arm,rgb_consumed=model.use_rgb,spatial_grid=[16,12],mode='offline_bidirectional',
        note='Candidate annotations for review; hand ROIs/camera calibration required. Uncertainty is empirical under synthetic partial occlusion, not guaranteed for natural occlusion.',
        tracks=[complete(t,model,encoder,probe,calibration,a.device) for t in obj['tracks']])
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(dest),frames=sum(len(t['frames']) for t in result['tracks']),arm=a.arm)))

if __name__=='__main__':main()
