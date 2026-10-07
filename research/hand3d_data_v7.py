import json,hashlib
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
import spatial_rgb_common as s
from hand3d_temporal_v7 import WRIST
RUN=s.common.ROOT/'experiments/offline_hand3d_v7';RUN.mkdir(exist_ok=True)
V4=s.common.ROOT/'experiments/natural_reliability_v4'

def save(path,obj):s.save(path,obj)

def camera_pose(camera):
    p=camera['T_world_from_camera'];q=p['quaternion_wxyz']
    return Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix().astype(np.float32),np.asarray(p['translation_xyz'],np.float32)

def prepare():
    records,index=s.records_and_index();rows=json.loads((s.OLD/'rows.json').read_text());old=torch.load(s.OLD/'windows.pt',weights_only=False)
    sampler=s.common.ROOT/'experiments/temporal_sampling_v6/multiscale_windows.pt'
    sampled=torch.load(sampler,weights_only=False);n=len(records)+1
    pieces={}
    for source in sorted({r['source'] for r in records}):
        data={};last=0
        for path in sorted((s.common.ROOT/'experiments'/source/'locked_observations').glob('*.pt')):
            c=torch.load(path,weights_only=False,mmap=True);assert c['start']==last;last=c['end']
            for key in ['wilor'] :data.setdefault(key,[]).append(c[key].float())
        pieces[source]={key:torch.cat(v) for key,v in data.items()}
    xyz_camera=torch.zeros(n,20,3);gt_camera=torch.zeros(n,20,3);gt_valid=torch.zeros(n,20,dtype=torch.bool)
    rotation=torch.eye(3)[None].repeat(n,1,1);translation=torch.zeros(n,3);scores=torch.zeros(n)
    for i,r in enumerate(records,1):
        xyz_camera[i]=pieces[r['source']]['wilor'][r['index']];R,t=camera_pose(r['camera']);rotation[i]=torch.from_numpy(R);translation[i]=torch.from_numpy(t);scores[i]=r['score']
        if r['matched']:
            gt_camera[i]=torch.tensor(r['gt']);gt_valid[i]=torch.isfinite(gt_camera[i]).all(-1)
    available=torch.isfinite(xyz_camera).all(-1);available[0]=False;xyz_camera=torch.nan_to_num(xyz_camera)
    world=torch.einsum('njc,nkc->njk',xyz_camera,rotation)+translation[:,None]
    rgb=torch.zeros(n,192,128,dtype=torch.float16);positions=torch.zeros(n,192,2);rays_camera=torch.zeros(n,192,3);last=0
    yy,xx=np.mgrid[:16,:12];canonical=np.stack([xx*16+37.5,yy*16+5.5],-1).reshape(192,2)
    for path in sorted((s.RUN/'spatial_chunks').glob('*.pt')):
        c=torch.load(path,weights_only=False,mmap=True);assert c['start']==last;last=c['end'];lo,hi=c['start']+1,c['end']+1
        rgb[lo:hi]=c['features'][:,0];positions[lo:hi]=c['positions']
        focal=c['focal'].float();uv=torch.from_numpy(canonical.astype(np.float32))[None]
        rays=torch.cat([(uv-127.5)/focal[:,None,None],torch.ones(hi-lo,192,1)],-1)
        rays_camera[lo:hi]=torch.einsum('nsc,nkc->nsk',rays,c['transform'].float())
    assert last==len(records)
    rays_camera=rays_camera/rays_camera.norm(dim=-1,keepdim=True).clamp_min(1e-6)
    rays_world=torch.einsum('nsc,nkc->nsk',rays_camera,rotation)
    roles=json.loads((V4/'protocol.json').read_text())['splits'];risk_cache=torch.load(V4/'features.pt',weights_only=False,mmap=True)
    center=sampled['feature_ids'][:,8];gt=gt_camera[center];valid=gt_valid[center]
    # Center labels and 3D observation lineage must match the existing fixed rows.
    for j,r in enumerate(rows):
        rec=records[int(center[j])-1];assert rec['source']==r['source'] and rec['index']==r['row_index']
    reconstructed=torch.einsum('njc,nck->njk',world-translation[:,None],rotation)
    roundtrip=float((reconstructed[1:]-xyz_camera[1:]).abs().max());assert roundtrip<2e-6,roundtrip
    data=dict(feature_ids=sampled['feature_ids'],dt=sampled['dt'],xy=sampled['xy'],observed_2d=sampled['observed'],world=world,xyz_camera_bank=xyz_camera,available_bank=available,
        rotation=rotation,translation=translation,rays_world=rays_world,rgb_bank=rgb,positions_bank=positions,roi=index['roi'],scores=scores,risk_rgb_bank=risk_cache['frame_bank'],
        gt=gt,valid=valid,gt_uv=old['gt'],uv_valid=old['valid'],roles=risk_cache['roles'],subjects=[r['subject'] for r in rows],rows=rows)
    assert all(torch.isfinite(v).all() for k,v in data.items() if torch.is_tensor(v) and v.is_floating_point())
    torch.save(data,RUN/'data.pt')
    save(RUN/'data_checks.json',dict(camera_world_roundtrip_max_m=roundtrip,windows=len(rows),observations=len(records),center_3d_available_fraction=float(available[center].float().mean()),
        gt_center_valid_fraction=float(valid.float().mean()),roles=roles,align='Every historical/future point is transformed eye->world->current eye using camera extrinsics; no GT in transforms',
        sampler='17 nonuniform time slots: nearest 1/6s, farthest +/-4s; existing predicted tracks, no re-association',rgb='192 spatial cells retained; frozen full WiLoR and previously trained spatial stem',
        source_3d='native frozen WiLoR camera XYZ, not depth inferred from corrected 2D',gt_valid='finite 3D annotations; projection visibility is not used as 3D-validity truth'))
    print(json.dumps(dict(prepared=True,windows=len(rows),roundtrip_m=roundtrip)))

def load(device):
    raw=torch.load(RUN/'data.pt',weights_only=False,mmap=True)
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()}

def batch(data,ids,risk=None,confirmed=None):
    f=data['feature_ids'][ids];center=f[:,8];R=data['rotation'][center];t=data['translation'][center]
    xyz=torch.einsum('btjc,bck->btjk',data['world'][f]-t[:,None,None],R)
    exists=data['available_bank'][f]&(f>0)[...,None];xyz=torch.where(exists[...,None],xyz,0.)
    rays=torch.einsum('btsc,bck->btsk',data['rays_world'][f],R)
    origin=torch.einsum('btc,bck->btk',data['translation'][f]-t[:,None],R)*(f>0)[...,None]
    if confirmed is None:confirmed=torch.zeros(len(ids),20,dtype=torch.bool,device=ids.device)
    if risk is None:camera=relative=torch.zeros(len(ids),20,device=ids.device)
    else:camera=risk[ids,:,0];relative=risk[ids,:,1]
    # Explicit observation-only whitelist; targets remain outside this dict.
    safe_xy=torch.where(data['observed_2d'][ids][...,None],data['xy'][ids],0.)
    return dict(xyz=xyz,available=exists,base=data['xyz_camera_bank'][center],dt=data['dt'][ids],xy=safe_xy,observed_2d=data['observed_2d'][ids],
        rgb=data['rgb_bank'][f],positions=data['positions_bank'][f],roi=data['roi'][f],scores=data['scores'][f],rays=rays,camera_origin=origin,
        rgb_valid=f>0,confirmed=confirmed,risk_camera=camera,risk_relative=relative,risk_rgb=data['risk_rgb_bank'][f])

def risk_features(b):
    xyz=b['xyz'];available=b['available'];base=b['base'];root=base[:,WRIST:WRIST+1];pose=base-root;B=len(base)
    # Interpolation excludes the central observation. All times are real offsets.
    parts=[base/.5,pose/.1,available[:,8,:,None].float(),b['observed_2d'][:,8,:,None].float(),b['scores'][:,8,None,None].expand(-1,20,1),torch.eye(20,device=base.device)[None].expand(B,-1,-1)]
    for radius in [0,1,3]:
        mask=available.clone();mask[:,8-radius:9+radius]=False
        index=torch.arange(17,device=base.device)[None,:,None].expand(B,17,20)
        left=torch.where(mask&(b['dt'][:,:,None]<=0),index,-1).amax(1);right=torch.where(mask&(b['dt'][:,:,None]>=0),index,17).amin(1)
        li=left.clamp_min(0);ri=right.clamp_max(16);rows=torch.arange(B,device=base.device)[:,None];joints=torch.arange(20,device=base.device)[None]
        a=xyz[rows,li,joints];z=xyz[rows,ri,joints];ta=b['dt'].gather(1,li);tz=b['dt'].gather(1,ri)
        w=(-ta/(tz-ta).clamp_min(1e-5)).clamp(0,1);both=(left>=0)&(right<17);anyok=(left>=0)|(right<17)
        estimate=a*(1-w[...,None])+z*w[...,None];nearest=torch.where((left>=0)[...,None],a,z);estimate=torch.where(both[...,None],estimate,nearest)
        parts.extend([((base-estimate)/.1).clamp(-10,10)*anyok[...,None],both[...,None].float(),anyok[...,None].float()])
    size=(b['roi'][:,8,2:]-b['roi'][:,8,:2]).mean(-1).clamp_min(.01)
    for t in [5,8,11]:
        dist=((b['positions'][:,t,None]-b['xy'][:,t,:,None])/size[:,None,None,None]).square().sum(-1)
        w=(-dist/(2*(1/12)**2)).softmax(-1)
        parts.append(torch.einsum('bjs,bsc->bjc',w,b['risk_rgb'][:,t].float())*b['rgb_valid'][:,t,None,None])
    parts.append(b['risk_rgb'][:,8].float().mean(1)[:,None].expand(-1,20,-1))
    x=torch.cat(parts,-1);assert torch.isfinite(x).all();return x

if __name__=='__main__':prepare()
