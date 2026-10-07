"""Offline RGB + 2D trajectory completion. Images and hand boxes are required.

Tracks/frames schema extends v1 with image and box_xyxy. Optional
occluder_crop_xyxy is a synthetic rectangle in the 256x256 crop, applied before
the encoder; it also forces ALL coordinates in that frame to unobserved.
"""
import argparse,json,hashlib,sys
from pathlib import Path
import numpy as np
import torch

def complete(track,model,encoder,calibration,device):
    import cv2
    from offline_rgb_encoder import crop_roi,crop_image,cover,input_tensor
    from offline_kp_model import condition
    frames=track['frames'];times=np.array([f['timestamp_s'] for f in frames],float)
    assert len(times)>0 and np.all(np.diff(times)>0)
    xy=np.zeros((len(frames),20,2),np.float32);valid=np.zeros((len(frames),20),bool);rois=[];crops=[]
    for i,f in enumerate(frames):
        assert len(f['xy_px'])==20 and len(f['observed'])==20
        roi=crop_roi(f['box_xyxy']);rois.append(roi/1408)
        if encoder is not None:
            image=cv2.imread(f['image']);assert image is not None,f['image'];assert image.shape[:2]==(1408,1408)
            crop=crop_image(image,roi)
            if f.get('occluder_crop_xyxy') is not None:crop=cover(crop,f['occluder_crop_xyxy'],f.get('occluder_color',128))
            crops.append(crop)
        rect=f.get('occluder_crop_xyxy');pixel_mask=rect is not None and rect[2]>rect[0] and rect[3]>rect[1]
        for j,(p,seen) in enumerate(zip(f['xy_px'],f['observed'])):
            if seen and not pixel_mask:
                assert p is not None and np.isfinite(p).all();xy[i,j]=np.asarray(p)/1408;valid[i,j]=True
    features=[]
    if encoder is not None:
        for start in range(0,len(crops),96):
            part=crops[start:start+96];x=input_tensor(part,device);n=len(x)
            if n<96:x=torch.cat([x,torch.zeros(96-n,3,256,256,device=device)])
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):z=encoder(x)
            features.append(z[:n].float().half().cpu())
        bank=torch.cat(features)
    else:bank=torch.zeros(len(frames),16,512,dtype=torch.float16)
    rois=np.asarray(rois);windows=[];masks=[];dts=[];context_ids=[];context_roi=[];context_valid=[]
    for now in times:
        wx=np.zeros((17,20,2),np.float32);wm=np.zeros((17,20),bool);dt=np.arange(-8,9,dtype=np.float32)/6
        ci=np.zeros(17,np.int64);cr=np.zeros((17,4),np.float32);cv=np.zeros(17,bool)
        for k,target in enumerate(now+np.arange(-8,9)/6):
            idx=int(np.abs(times-target).argmin())
            if abs(times[idx]-target)>1/12:continue
            wx[k]=xy[idx];wm[k]=valid[idx];dt[k]=times[idx]-now;ci[k]=idx;cr[k]=rois[idx];cv[k]=True
        windows.append(wx);masks.append(wm);dts.append(dt);context_ids.append(ci);context_roi.append(cr);context_valid.append(cv)
    results=[]
    for start in range(0,len(frames),96):
        sl=slice(start,start+96)
        b=condition(torch.tensor(np.stack(windows[sl]),device=device),torch.tensor(np.stack(masks[sl]),device=device),torch.tensor(np.stack(dts[sl]),device=device))
        cv=torch.tensor(np.stack(context_valid[sl]),device=device)
        rgb=bank[np.stack(context_ids[sl])].to(device)*cv[:,:,None,None]
        b.update(rgb=rgb,roi=torch.tensor(np.stack(context_roi[sl]),device=device),rgb_valid=cv)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):out=model.predict(b,samples=4)
        p=out['xy'].cpu().numpy()*1408;draws=out['samples'].cpu().numpy()*1408;std=out['std'].norm(dim=-1).cpu().numpy()*1408
        context=out['has_context'].cpu().numpy();missing=b['missing'].cpu().numpy()
        for k in range(len(p)):
            i=start+k;points=[]
            for j in range(20):
                inferred=bool(missing[k,j]);available=bool(context[k,j])
                value=p[k,j].tolist() if available else None
                if not inferred:value=frames[i]['xy_px'][j]
                radius=float(calibration['factor']*(std[k,j]+calibration['floor_px'])) if calibration and inferred and available else None
                points.append(dict(xy_px=value,source='inferred' if inferred else 'input',available=available,
                    candidate_xy_px=draws[:,k,j].tolist() if inferred and available else [],
                    empirical_radius90_px=radius,review_required=True,reviewed=False))
            results.append(dict(timestamp_s=frames[i]['timestamp_s'],image=frames[i]['image'],points=points))
    return dict(id=track.get('id'),frames=results)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--run',default='/mnt/why/HOT3D/experiments/offline_keypoint_rgb_v2');ap.add_argument('--arm',choices=['rgb_dit','rgb_regression','tracks_dit','tracks_regression'],default='rgb_dit');ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    torch.set_num_threads(4);run=Path(a.run).resolve();sys.path.insert(0,str(run/'sealed'))
    from offline_rgb_model import RGBKeypointCompleter
    from offline_rgb_adapter import RGBAdapterCompleter
    from offline_rgb_encoder import RGBEncoder,WEIGHT
    obj=json.loads(Path(a.input).read_text());assert obj['image_size']==[1408,1408]
    seal=json.loads((run/'sealed/selection.json').read_text());checkpoint=run/'sealed'/f'{a.arm}.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==seal['models'][a.arm]
    for name,digest in seal['code'].items():assert hashlib.sha256((run/'sealed'/name).read_bytes()).hexdigest()==digest,name
    ck=torch.load(checkpoint,map_location=a.device,weights_only=False)
    cls=RGBAdapterCompleter if ck.get('architecture')=='adapter' else RGBKeypointCompleter
    model=cls(ck['kind'],ck['use_rgb']).to(a.device).eval();model.load_state_dict(ck['model'])
    encoder=RGBEncoder().to(a.device).eval() if ck['use_rgb'] else None
    if encoder is not None:assert hashlib.sha256(WEIGHT.read_bytes()).hexdigest()==seal['encoder_sha256']
    calibration=json.loads((run/'sealed/uncertainty.json').read_text()) if a.arm=='rgb_dit' else None
    result=dict(image_size=obj['image_size'],arm=a.arm,mode='offline_bidirectional_rgb_and_tracks' if ck['use_rgb'] else 'offline_tracks_and_boxes',
        rgb_consumed=ck['use_rgb'],note='Candidate annotations for manual review; supplied hand boxes remain available. Empirical uncertainty calibration is from artificial partial occlusion.',
        tracks=[complete(t,model,encoder,calibration,a.device) for t in obj['tracks']])
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(dest),arm=a.arm,frames=sum(len(t['frames']) for t in result['tracks']),rgb_consumed=ck['use_rgb'])))

if __name__=='__main__':main()
