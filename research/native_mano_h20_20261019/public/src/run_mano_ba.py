"""Public fixed-camera MANO BA. Complete timeline, explicit wrist state, no GT."""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import h5py
from scipy.spatial.transform import Rotation
import os
ROOT=Path(os.environ.get('PUBLIC_HEADHAND_ROOT','/mnt/why/public_headhand_h20_20261019'))
sys.path.insert(0,str(ROOT/'vendor/Ego-Hand/research'))
import wilor_eval_common as common
from mano_model import load_decoder
from geometry import load_frames, camera
from hand3d_rollout_v8 import project_fisheye624
MAPPING=[4,8,12,16,20,0,2,3,5,6,7,9,10,11,13,14,15,17,18,19]

def tensor(value):return torch.as_tensor(np.asarray(value),device='cuda',dtype=torch.float32)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dit',action='store_true');ap.add_argument('--iterations',type=int,default=100)
    ap.add_argument('--suffix',default='');ap.add_argument('--head',default='head_clip2102');ap.add_argument('--depth',default='depth_fused_clip2102')
    args=ap.parse_args();torch.set_num_threads(4)
    out=ROOT/(('outputs/mano_dit_clip2102' if args.dit else 'outputs/mano_baseline_clip2102')+args.suffix);out.mkdir(exist_ok=False)
    frames=load_frames(ROOT/'data/clip-002102')
    rows=json.loads((ROOT/'outputs/wilor_clip2102/observations.json').read_text())['frames']
    head=json.loads((ROOT/f'outputs/{args.head}/headpose.json').read_text())['frames']
    coarse=np.load(ROOT/'outputs/wilor_clip2102/observations.npz')
    depth=json.loads((ROOT/f'outputs/{args.depth}/depth.json').read_text())
    N=len(rows);fi=np.array([r['frame_index'] for r in rows]);times=np.array([r['timestamp_ns'] for r in rows],np.int64)
    track=np.array([r['track_id'] for r in rows]);track_ids=sorted(set(track));group=np.array([track_ids.index(t) for t in track])
    right=coarse['right'].astype(bool).copy()
    for tid in track_ids:right[track==tid]=np.mean(right[track==tid])>=.5
    world_from_cam=np.array([head[i]['T_world_from_rgb'] for i in fi])
    R=tensor(world_from_cam[:,:3,:3]);T=tensor(world_from_cam[:,:3,3]);right_t=torch.tensor(right,device='cuda')
    decoder=load_decoder();initial=coarse['state'].copy();raw_initial=initial.copy()
    betas=np.stack([np.median(coarse['betas'][track==tid],axis=0) for tid in track_ids])
    anchor=np.zeros(N,np.float32);weights=np.zeros(N,np.float32);anchor_source=['unavailable']*N
    for tid in track_ids:
        known=sorted([x for x in depth['observations'] if x['track_id']==int(tid) and x['valid']],key=lambda x:x['timestamp_ns'])
        if not known:continue
        kt=np.array([x['timestamp_ns'] for x in known],np.int64);kz=np.array([x['wrist_depth_m'] for x in known])
        for i in np.flatnonzero(track==tid):
            nearest=int(np.argmin(np.abs(kt-times[i])));delta=abs(int(kt[nearest])-int(times[i]))/1e9
            if delta>.071:continue
            pos=np.searchsorted(kt,times[i]);bracket=0<pos<len(kt) and (kt[pos]-kt[pos-1])<100_000_000
            anchor[i]=np.interp(float(times[i]-kt[0]),(kt-kt[0]).astype(float),kz) if bracket else kz[nearest]
            mono='UniDepth' in known[nearest]['source'];weights[i]=.25 if mono else 1.
            anchor_source[i]=('interpolated-' if delta>1e-6 else '')+known[nearest]['source']
    valid_anchor=weights>0
    initial[valid_anchor,:3]*=(anchor[valid_anchor]/initial[valid_anchor,2])[:,None]
    init_cam=tensor(initial);beta0=tensor(betas);group_t=torch.tensor(group,device='cuda')
    params=tensor([[*camera(frames[i]['cameras']['214-1']['calibration']).f,
                    *camera(frames[i]['cameras']['214-1']['calibration']).c,
                    *camera(frames[i]['cameras']['214-1']['calibration']).distort] for i in fi])
    uv=tensor([r['xy'] for r in rows]);scale=tensor([max(r['box_xyxy'][2]-r['box_xyxy'][0],r['box_xyxy'][3]-r['box_xyxy'][1]) for r in rows])[:,None,None]
    with torch.no_grad():
        test=project_fisheye624(tensor(coarse['xyz_camera']),params).cpu().numpy()
    reference=np.stack([camera(frames[f]['cameras']['214-1']['calibration']).eye_to_window(xyz) for f,xyz in zip(fi,coarse['xyz_camera'])])
    projection_error=float(np.max(np.abs(test-reference)));assert projection_error<.02,projection_error
    fit_details=None
    if args.dit:
        proposals=json.loads((ROOT/f'outputs/dit_clip2102{args.suffix}/proposals20.json').read_text())
        lookup={(int(t['id']),int(round(f['timestamp_s']*1e9))):f for t in proposals['tracks'] for f in t['frames']}
        target=tensor([lookup[(int(track[i]),int(times[i]))]['candidate_xyz_camera_m'] for i in range(N)])
        target=target-target[:,5:6]
        pose=torch.nn.Parameter(init_cam[:,3:].clone());fixed=init_cam[:,:3].detach()
        optimizer=torch.optim.LBFGS([pose],lr=.6,max_iter=70,line_search_fn='strong_wolfe',history_size=15)
        fit_calls=[0]
        def fit_closure():
            optimizer.zero_grad();j=decoder(torch.cat([fixed,pose],-1),beta0[group_t],right_t)
            relative=j[:,MAPPING]-j[:,:1]
            loss=torch.nn.functional.smooth_l1_loss(relative/.02,target/.02)+.02*(pose-init_cam[:,3:]).square().mean()
            loss.backward();fit_calls[0]+=1;return loss
        before=float(fit_closure().detach());optimizer.step(fit_closure);after=float(fit_closure().detach())
        candidate=torch.cat([fixed,pose.detach()],-1)
        with torch.no_grad():
            a=decoder(init_cam,beta0[group_t],right_t);b=decoder(candidate,beta0[group_t],right_t)
            cost_a=torch.nn.functional.smooth_l1_loss((project_fisheye624(a,params)-uv)/scale,torch.zeros_like(uv),reduction='none').mean((1,2))
            cost_b=torch.nn.functional.smooth_l1_loss((project_fisheye624(b,params)-uv)/scale,torch.zeros_like(uv),reduction='none').mean((1,2))
        # Preserve original candidate; a track-wide choice avoids arbitrary per-frame switching.
        accepted=[]
        for tid in track_ids:
            ids=np.flatnonzero(track==tid);idx=torch.tensor(ids,device='cuda')
            def motion_cost(j):
                rel=j[idx]-j[idx,:1];dt=tensor(np.diff(times[ids])/1e9)[:,None,None]
                v=(rel[1:]-rel[:-1])/(60*dt)
                return float((v[1:]-v[:-1]).square().mean())
            ca=float(cost_a[idx].mean())+.2*motion_cost(a);cb=float(cost_b[idx].mean())+.2*motion_cost(b)
            use=cb<=ca+.0001
            if use:init_cam[idx]=candidate[idx]
            accepted.append(dict(track_id=int(tid),original_cost=ca,dit_cost=cb,selected='dit' if use else 'original'))
        fit_details=dict(semantic_bridge='20 target points fitted to 21-point MANO; wrist fixed; shape fixed',
                         objective_before=before,objective_after=after,evaluations=fit_calls[0],selection=accepted,
                         GT_used=False,selection_rule='track-level robust WiLoR-2D agreement plus relative-motion cost; conservative 0.0001 cost margin')
        (out/'dit_fit.json').write_text(json.dumps(fit_details,indent=2))
    camera_seed=init_cam.detach().cpu().numpy()
    initial_world=camera_seed.copy()
    initial_world[:,:3]=np.einsum('nij,nj->ni',world_from_cam[:,:3,:3],camera_seed[:,:3])+world_from_cam[:,:3,3]
    initial_world[:,3:6]=Rotation.from_matrix(world_from_cam[:,:3,:3]@Rotation.from_rotvec(camera_seed[:,3:6]).as_matrix()).as_rotvec()
    state=torch.nn.Parameter(tensor(initial_world));shape=torch.nn.Parameter(beta0.clone());reference_state=state.detach().clone()
    anchor_t=tensor(anchor);weight_t=tensor(weights)
    pairs=[];triples=[]
    for tid in track_ids:
        ids=np.flatnonzero(track==tid)
        for a,b in zip(ids[:-1],ids[1:]):
            if 0<(times[b]-times[a])<100_000_000 and head[fi[a]]['segment']==head[fi[b]]['segment']:pairs.append((a,b))
        for a,b,c in zip(ids[:-2],ids[1:-1],ids[2:]):
            if 0<(times[c]-times[b])<100_000_000 and 0<(times[b]-times[a])<100_000_000 and head[fi[a]]['segment']==head[fi[c]]['segment']:triples.append((a,b,c))
    pair=torch.tensor(pairs,device='cuda');triple=torch.tensor(triples,device='cuda')
    dp=tensor([(times[b]-times[a])/1e9 for a,b in pairs])[:,None,None]
    dt0=tensor([(times[b]-times[a])/1e9 for a,b,c in triples])[:,None,None]
    dt1=tensor([(times[c]-times[b])/1e9 for a,b,c in triples])[:,None,None]
    opt=torch.optim.LBFGS([state,shape],lr=.7,max_iter=args.iterations,line_search_fn='strong_wolfe',history_size=15)
    history=[];start=time.time()
    def evaluate():
        world=decoder(state,shape[group_t],right_t);cam=torch.einsum('nji,nkj->nki',R,world-T[:,None])
        reproj=torch.nn.functional.smooth_l1_loss((project_fisheye624(cam,params)-uv)/scale,torch.zeros_like(uv))
        depth_loss=(torch.nn.functional.smooth_l1_loss((cam[:,0,2]-anchor_t)/.02,torch.zeros_like(anchor_t),reduction='none')*weight_t).sum()/weight_t.sum().clamp_min(1)
        prior=(state[:,6:]-reference_state[:,6:]).square().mean();root_prior=(state[:,3:6]-reference_state[:,3:6]).square().mean()
        position_prior=((state[:,:3]-reference_state[:,:3])/.1).square().mean()
        v=(world[pair[:,1]]-world[pair[:,0]])/(60*dp)
        v0=(world[triple[:,1]]-world[triple[:,0]])/(60*dt0)
        v1=(world[triple[:,2]]-world[triple[:,1]])/(60*dt1)
        a=(v1-v0)/(60*(dt0+dt1)/2)
        speed=torch.nn.functional.smooth_l1_loss(v/.01,torch.zeros_like(v))
        accel=torch.nn.functional.smooth_l1_loss(a/.005,torch.zeros_like(a))
        # SO(3) chordal increments, no invalid interpolation of rotation matrices.
        from smplx.lbs import batch_rodrigues
        rotations=batch_rodrigues(state[:,3:6])
        angular=((rotations[pair[:,1]]-rotations[pair[:,0]])/(60*dp)).square().mean()
        total=20*reproj+5*depth_loss+.08*prior+.05*root_prior+.01*position_prior+.002*speed+.02*accel+.002*angular+.02*(shape-beta0).square().mean()
        return total,world,cam,dict(reprojection=float(reproj.detach()),depth=float(depth_loss.detach()),pose_prior=float(prior.detach()),acceleration=float(accel.detach()))
    def closure():
        opt.zero_grad();loss,_,_,parts=evaluate();assert torch.isfinite(loss);loss.backward()
        assert torch.isfinite(state.grad).all() and torch.isfinite(shape.grad).all()
        history.append(dict(evaluation=len(history),loss=float(loss.detach()),**parts))
        if len(history)%25==0:print(json.dumps(history[-1]),flush=True)
        return loss
    before=float(closure().detach());opt.step(closure)
    with torch.no_grad():loss,world,cam,parts=evaluate()
    world=world.cpu().numpy();cam=cam.cpu().numpy();state_np=state.detach().cpu().numpy();shape_np=shape.detach().cpu().numpy()
    assert np.isfinite(world).all() and (cam[:,:,2]>.01).all()
    uv_end=project_fisheye624(tensor(cam),params).cpu().numpy()
    reproj_px=float(np.linalg.norm(uv_end-uv.cpu().numpy(),axis=-1).mean())
    np.savez_compressed(out/'parameters.npz',state_world51=state_np,betas_shared=shape_np,group=group,right=right,
                        joints_world=world,joints_camera=cam,frame_index=fi,track_id=track,timestamp_ns=times)
    result_frames=[]
    for fidx,frame in enumerate(frames):
        hands=[]
        for i in np.flatnonzero(fi==fidx):
            rootR=Rotation.from_rotvec(state_np[i,3:6]);hands.append(dict(track_id=int(track[i]),right=bool(right[i]),owner='unknown',valid=True,
                joints_world_m=world[i].tolist(),joints_camera_m=cam[i].tolist(),wrist_world_m=state_np[i,:3].tolist(),
                root_quaternion_xyzw=rootR.as_quat().tolist(),local_axis_angle_rad=state_np[i,6:].reshape(15,3).tolist(),
                shape10=shape_np[group[i]].tolist(),depth_anchor_source=anchor_source[i],depth_anchor_m=float(anchor[i]) if weights[i]>0 else None,
                review_required=True))
        result_frames.append(dict(timestamp_ns=frame['timestamp_ns'],frame=frame['id'],head_valid=head[fidx]['valid'],
                                  camera_segment=head[fidx]['segment'],T_world_from_camera=head[fidx]['T_world_from_rgb'],hands=hands))
    payload=dict(complete=True,GT_used=False,algorithm='public fixed-camera MANO BA',dit=args.dit,frames=result_frames,
        skeleton='MANO21 OpenPose order, wrist0, meters; state51 begins with wrist XYZ, not MANO transl',
        shapes_shared='within predicted track only; owner unknown',motion='real per-pair and per-triple timestamps, 60Hz reference; no crossing reset/gap>.1s',
        observation='WiLoR projected 2D, not independent image keypoint labels',
        constraints='MANO geometric FK, pose/shape priors, soft motion; no hard speed, acceleration, collision or anatomical guarantee',
        production_parity=False,dit_fit=fit_details,head_source=args.head,depth_source=args.depth,
        experiment_scope='development repair after first pilot evaluation' if args.suffix else 'initial diagnostic pilot')
    (out/'joints3d_world.json').write_text(json.dumps(payload))
    with h5py.File(out/'bundle.h5','w') as file:
        file.create_dataset('timestamp_ns',data=np.array([f['timestamp_ns'] for f in frames],np.int64))
        file.create_dataset('T_world_from_camera',data=np.array([h['T_world_from_rgb'] for h in head]))
        for tid in track_ids:
            g=file.create_group(f'track_{tid}');values=np.full((len(frames),21,3),np.nan,np.float32);valid=np.zeros(len(frames),bool)
            ids=np.flatnonzero(track==tid);values[fi[ids]]=world[ids];valid[fi[ids]]=True
            g.create_dataset('joints_world_m',data=values);g.create_dataset('valid',data=valid)
    report=dict(complete=True,GT_used=False,frames=len(frames),hand_observations=N,tracks=len(track_ids),projection_parity_max_px=projection_error,
                objective_before=before,objective_after=float(loss),reprojection_mean_px=reproj_px,seconds=time.time()-start,
                depth_supported=int(valid_anchor.sum()),evaluations=len(history),loss_history=history)
    (out/'optimization.json').write_text(json.dumps(report,indent=2))
    seals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}
    (out/'seal.json').write_text(json.dumps(dict(files=seals,GT_used=False),indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='loss_history'}),flush=True)

if __name__=='__main__':main()
