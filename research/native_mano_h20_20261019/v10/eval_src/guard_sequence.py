"""Enforce existing displacement/reprojection gates after BA in MANO space.

One scalar per complete track; geodesic rotations and shared shape. No GT,
per-point clipping, per-frame switches or relaxation of the 5/10/30/.2 gates.
"""
import argparse,copy,hashlib,json,os,subprocess,sys
from pathlib import Path
import h5py,numpy as np,torch
from scipy.spatial.transform import Rotation
from native_mano import NativeMANO
sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/src')
from geometry import camera,load_frames
from hand3d_rollout_v8 import project_fisheye624
R=Path('/mnt/why/native_candidate_sequence_v10_20261019');VIEW=R/'full_pipeline';SOURCE=Path('/mnt/why/public_headhand_h20_20261019/src')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['frame_RGB', 'temporal_noRGB', 'temporal_RGB', 'temporal_noRGB_scaled', 'temporal_RGB_scaled'],required=True);a=ap.parse_args()
    torch.set_num_threads(4);out=VIEW/f'outputs/ba_{a.arm}_guarded';out.mkdir(exist_ok=False)
    basepath=VIEW/'outputs/mano_baseline_clip2102';proppath=VIEW/f'outputs/ba_{a.arm}'
    base=np.load(basepath/'parameters.npz');prop=np.load(proppath/'parameters.npz');frames=load_frames(VIEW/'data/clip-002102')
    rows=json.loads((VIEW/'outputs/wilor_clip2102/observations.json').read_text())['frames'];head=json.loads((VIEW/'outputs/head_clip2102/headpose.json').read_text())['frames']
    for name in ['timestamp_ns','track_id','right','group','frame_index']:assert np.array_equal(base[name],prop[name])
    tensor=lambda x:torch.as_tensor(np.asarray(x),device='cuda',dtype=torch.float32)
    codec=NativeMANO();right=torch.tensor(base['right'],device='cuda');group=base['group'];state=base['state_world51'].copy();shapes=base['betas_shared'].copy()
    poses=np.array([head[i]['T_world_from_rgb'] for i in base['frame_index']]);rot=tensor(poses[:,:3,:3]);trans=tensor(poses[:,:3,3])
    params=tensor([[*camera(frames[i]['cameras']['214-1']['calibration']).f,*camera(frames[i]['cameras']['214-1']['calibration']).c,*camera(frames[i]['cameras']['214-1']['calibration']).distort] for i in base['frame_index']]);uv=tensor([r['xy'] for r in rows])
    final_world=np.empty_like(base['joints_world']);final_cam=np.empty_like(base['joints_camera']);decisions=[]
    with torch.inference_mode():
        for g in np.unique(group):
            ids=np.flatnonzero(group==g);s0=base['state_world51'][ids];s1=prop['state_world51'][ids]
            a0=Rotation.from_rotvec(s0[:,3:].reshape(-1,3));a1=Rotation.from_rotvec(s1[:,3:].reshape(-1,3));relative=(a0.inv()*a1).as_rotvec()
            reference=tensor(base['joints_camera'][ids]);reference_uv=project_fisheye624(reference,params[ids]);reproj0=float((reference_uv-uv[ids]).norm(dim=-1).mean())
            def at(alpha):
                s=s0.copy();s[:,:3]=(1-alpha)*s0[:,:3]+alpha*s1[:,:3]
                s[:,3:]=(a0*Rotation.from_rotvec(relative*alpha)).as_rotvec().reshape(-1,48)
                beta=(1-alpha)*base['betas_shared'][g]+alpha*prop['betas_shared'][g]
                world=codec.from_state(tensor(s),tensor(beta[None].repeat(len(ids),0)),right[ids]);cam=torch.einsum('nji,nkj->nki',rot[ids],world-trans[ids,None])
                displacement=(cam-reference).norm(dim=-1)*1000
                stats=dict(RMSE_mm=float(displacement.square().mean().sqrt()),P95_mm=float(torch.quantile(displacement,.95)),max_mm=float(displacement.max()),reprojection_increase_px=float((project_fisheye624(cam,params[ids])-uv[ids]).norm(dim=-1).mean())-reproj0)
                # Small inward numerical margin; the acceptance gate stays .2px.
                good=stats['RMSE_mm']<=5 and stats['P95_mm']<=10 and stats['max_mm']<=30 and stats['reprojection_increase_px']<=.1999
                return good,s,beta,world.cpu().numpy(),cam.cpu().numpy(),stats
            selected=None
            for alpha in np.linspace(1,0,33):
                values=at(float(alpha))
                if values[0]:selected=(float(alpha),values);break
            assert selected is not None
            lo,values=selected;hi=min(1,lo+1/32)
            for _ in range(14):
                mid=(lo+hi)/2;candidate=at(mid)
                if candidate[0]:lo=mid;values=candidate
                else:hi=mid
            _,s,beta,world,cam,stats=values;state[ids]=s;shapes[g]=beta;final_world[ids]=world;final_cam[ids]=cam
            decisions.append(dict(group=int(g),track=int(base['track_id'][ids[0]]),alpha=lo,**stats))
    data={k:base[k].copy() for k in base.files};data.update(state_world51=state,betas_shared=shapes,joints_world=final_world,joints_camera=final_cam)
    np.savez_compressed(out/'parameters.npz',**data)
    payload=json.loads((proppath/'joints3d_world.json').read_text());payload['post_BA_guard']=dict(GT_used=False,one_alpha_per_track=True,rotation='SO3 geodesic',shape='shared parameter interpolation',decisions=decisions)
    factors={x['group']:x['alpha'] for x in decisions}
    for i,fi in enumerate(base['frame_index']):
        hand=next(h for h in payload['frames'][fi]['hands'] if h['track_id']==int(base['track_id'][i]))
        hand.update(joints_world_m=final_world[i].tolist(),joints_camera_m=final_cam[i].tolist(),wrist_world_m=state[i,:3].tolist(),root_quaternion_xyzw=Rotation.from_rotvec(state[i,3:6]).as_quat().tolist(),local_axis_angle_rad=state[i,6:].reshape(15,3).tolist(),shape10=shapes[group[i]].tolist(),post_BA_guard_alpha=factors[int(group[i])])
    (out/'joints3d_world.json').write_text(json.dumps(payload))
    with h5py.File(out/'bundle.h5','w') as f:
        f['timestamp_ns']=np.array([x['timestamp_ns'] for x in frames],np.int64);f['T_world_from_camera']=np.array([h['T_world_from_rgb'] for h in head])
        for tid in np.unique(base['track_id']):
            ids=np.flatnonzero(base['track_id']==tid);g=f.create_group(f'track_{tid}');values=np.full((len(frames),21,3),np.nan,np.float32);valid=np.zeros(len(frames),bool);values[base['frame_index'][ids]]=final_world[ids];valid[base['frame_index'][ids]]=True;g['joints_world_m']=values;g['valid']=valid
    # Check the complete final output, including the numerical FK roundtrip.
    with torch.inference_mode():
        back=codec.from_state(tensor(state),tensor(shapes[group]),right);assert float((back-tensor(final_world)).abs().max())<1e-6
        pix=project_fisheye624(tensor(final_cam),params);reproj=float((pix-uv).norm(dim=-1).mean())
    delta=np.linalg.norm(final_cam-base['joints_camera'],axis=-1)*1000;reproj_ref=json.loads((basepath/'optimization.json').read_text())['reprojection_mean_px']
    guard=dict(complete=True,GT_used=False,one_alpha_per_track=True,decisions=decisions,RMSE_mm=float(np.sqrt(np.mean(delta**2))),P95_mm=float(np.quantile(delta,.95)),max_mm=float(delta.max()),reprojection_mean_px=reproj,reprojection_increase_px=reproj-reproj_ref,thresholds=dict(RMSE_mm=5,P95_mm=10,max_mm=30,reprojection_increase_px=.2),same_objective_reoptimized=False,safe_default_admitted=False)
    assert guard['RMSE_mm']<=5 and guard['P95_mm']<=10 and guard['max_mm']<=30 and guard['reprojection_increase_px']<=.2
    (out/'optimization.json').write_text(json.dumps(guard,indent=2));(out/'seal.json').write_text(json.dumps(dict(files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()},GT_used=False),indent=2))
    (R/'receipts'/f'guard_{a.arm}.json').write_text(json.dumps(guard,indent=2));print(json.dumps(guard),flush=True)

if __name__=='__main__':main()
