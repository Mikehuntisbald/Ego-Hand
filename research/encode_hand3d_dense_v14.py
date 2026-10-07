"""Original RGB / predicted XYZ adapter using real near-dense time slots."""
import numpy as np
import torch
from hand3d_data_v7 import camera_pose
from offline_rgb_encoder import crop_roi
import spatial_rgb_common as s

OFFSETS=[-120,-80,-50,-30,-10,-3,-2,-1,0,1,2,3,10,30,50,80,120]

def sample(z,device):
    times=z['times'];chosen=[];windows=[]
    for i,now in enumerate(times):
        ids=np.zeros(17,int);valid=np.zeros(17,bool);dt=np.array(OFFSETS,np.float32)/30
        ids[8]=i;valid[8]=True;used={i}
        for slot in sorted((k for k in range(17) if k!=8),key=lambda k:abs(OFFSETS[k])):
            target=now+OFFSETS[slot]/30;j=int(np.abs(times-target).argmin())
            if abs(times[j]-target)>1/60+2e-6 or j in used:continue
            ids[slot]=j;valid[slot]=True;dt[slot]=times[j]-now;used.add(j)
        chosen.append([int(j) if ok else None for j,ok in zip(ids,valid)])
        observed=z['available'][ids]&valid[:,None]
        windows.append(dict(ids=ids,dt=dt,rgb_valid=valid,xy=np.where(observed[...,None],z['xy'][ids],0.),observed_2d=observed,roi=z['roi'][ids]*valid[:,None],positions=z['positions'][ids]*valid[:,None,None],scores=z['scores'][ids]*valid))
    return {k:torch.from_numpy(np.stack([w[k] for w in windows])).to(device) for k in windows[0]},chosen

@torch.inference_mode()
def encode(track,encoder,probe,projection,device):
    frames=[];native=[];available=[];confirmed=[];rotations=[];translations=[];rays_world=[]
    crops=[];positions=[];rois=[];scores=[]
    yy,xx=np.mgrid[:16,:12];canonical=np.stack([xx*16+37.5,yy*16+5.5],-1).reshape(192,2)
    assert len(track['frames'])
    for f in track['frames']:
        xyz=np.asarray(f['xyz_camera_m'],np.float32);assert xyz.shape==(20,3)
        exists=np.asarray(f.get('available_3d',np.isfinite(xyz).all(-1)),bool)
        locks=np.asarray(f.get('confirmed_3d',[False]*20),bool)
        assert exists.shape==locks.shape==(20,) and np.all(~locks|exists) and np.isfinite(xyz[exists]).all()
        xyz=np.where(exists[:,None],xyz,0.)
        uv=np.asarray(f.get('xy_px',s.common.from_json(f['camera']).eye_to_window(xyz)),np.float32)
        valid2=np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<1408).all(-1)&exists
        frames.append(dict(image=f['image'],camera=f['camera'],timestamp_s=f['timestamp_s'],box_xyxy=f['box_xyxy'],box_confidence=f['box_confidence'],clip=f.get('clip',0),xy_px=np.nan_to_num(uv).tolist(),available=valid2.tolist(),confirmed=[False]*20))
        native.append(xyz);available.append(exists);confirmed.append(locks)
        R,t=camera_pose(f['camera']);rotations.append(R);translations.append(t)
        assert not f.get('occluder_crop_xyxy'),'Natural inference does not manufacture occlusions'
        score=float(f['box_confidence']);assert 0<=score<=1;scores.append(score)
        roi=crop_roi(f['box_xyxy']);ims,pos,transform,focal,*_=s.prepare(dict(image=f['image'],camera=f['camera'],clip=f.get('clip',0)),roi,[[0,0,0,0]])
        crops.append(ims[0]);positions.append(pos);rois.append(roi/1408)
        rays=np.c_[(canonical-127.5)/focal,np.ones(192)].astype(np.float32)@transform.T
        rays/=np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-6);rays_world.append(rays@R.T)
    times=np.array([f['timestamp_s'] for f in frames]);assert np.all(np.diff(times)>0)
    # Keep the pixel ROI exactly. A normalize/de-normalize float32 roundtrip
    # can change a crop boundary pixel and amplify through the visual encoder.
    # Match cache BF16 projection and padded16-frame batches on the same device.
    dense=[];risk_rgb=[];spatial=[];projector=projection.to(device)
    for start in range(0,len(crops),16):
        part=crops[start:start+16];n=len(part);part=part+[part[-1]]*(16-n)
        inp=s.common.input_tensor(part,[1]*16,rotation=1).to(device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            feature=encoder(inp[:,:,:,32:-32])[-1].flatten(2).transpose(1,2)
            projected=feature.float()@projector.float()
            localized=probe.features(feature).flatten(2).transpose(1,2)
        dense.append(feature[:n].half().cpu())
        risk_rgb.append(projected[:n].half().cpu());spatial.append(localized[:n].half().cpu())
    z=dict(times=times,xy=np.array([f['xy_px'] for f in frames],np.float32)/1408,available=np.array([f['available'] for f in frames],bool),positions=np.stack(positions),roi=np.stack(rois),scores=np.array(scores,np.float32),native=torch.cat(dense),risk_rgb=torch.cat(risk_rgb),spatial=torch.cat(spatial))
    sampled,chosen=sample(z,device)
    xyz=torch.tensor(np.stack(native),device=device);exists=torch.tensor(np.stack(available),device=device);locks=torch.tensor(np.stack(confirmed),device=device)
    R=torch.tensor(np.stack(rotations),device=device);t=torch.tensor(np.stack(translations),device=device);rw=torch.tensor(np.stack(rays_world),device=device)
    ids=sampled['ids'];slots=sampled['rgb_valid'];world=torch.einsum('njc,nkc->njk',xyz,R)+t[:,None]
    aligned=torch.einsum('ntjc,nck->ntjk',world[ids]-t[:,None,None],R);aligned_exists=exists[ids]&slots[...,None]
    aligned=torch.where(aligned_exists[...,None],aligned,0.)
    rays=torch.einsum('ntsc,nck->ntsk',rw[ids],R)*slots[:,:,None,None]
    origin=torch.einsum('ntc,nck->ntk',t[ids]-t[:,None],R)*slots[...,None]
    b=dict(xyz=aligned,available=aligned_exists,base=xyz,dt=sampled['dt'],xy=sampled['xy'],observed_2d=sampled['observed_2d'],rgb=z['spatial'][ids.cpu()].to(device)*slots[:,:,None,None],positions=sampled['positions'],roi=sampled['roi'],scores=sampled['scores'],rays=rays,camera_origin=origin,rgb_valid=slots,confirmed=locks,risk_rgb=z['risk_rgb'][ids.cpu()].to(device)*slots[:,:,None,None],rgb_native=z['native'][ids.cpu()].to(device)*slots[:,:,None,None])
    return b,chosen
