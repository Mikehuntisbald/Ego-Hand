"""Native MANO inference on generic public-pipeline observations, no GT or IK."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
from native_dit import NativeMANODiT
from native_mano import NativeMANO,exp_so3
from train_native import predict

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public-root',type=Path,required=True);ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--protect',action='store_true');a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
    ck=torch.load(a.checkpoint,weights_only=False,map_location='cuda');model=NativeMANODiT(**ck['config']).cuda().eval();model.load_state_dict(ck['model']);codec=NativeMANO()
    coarse=np.load(a.public_root/'outputs/wilor_clip2102/observations.npz');rows=json.loads((a.public_root/'outputs/wilor_clip2102/observations.json').read_text())['frames']
    head=json.loads((a.public_root/'outputs/head_clip2102/headpose.json').read_text())['frames'];N=len(rows)
    # All values below are predicted observations / provided physical calibration.
    state=torch.tensor(coarse['state'],device='cuda');beta=torch.tensor(coarse['betas'],device='cuda');right=torch.tensor(coarse['right'],device='cuda');rgb=torch.tensor(coarse['rgb_spatial'],device='cuda')
    scores=torch.tensor([r['confidence'] for r in rows],device='cuda')
    poses=np.array([head[r['frame_index']]['T_world_from_rgb'] for r in rows]);R=torch.tensor(poses[:,:3,:3],device='cuda',dtype=torch.float32);T=torch.tensor(poses[:,:3,3],device='cuda',dtype=torch.float32)
    from temporal_window_v44 import SHORT_OFFSETS
    windows=[];dt=[];valid=[]
    for i,r in enumerate(rows):
        ids=np.array([j for j,x in enumerate(rows) if x['track_id']==r['track_id'] and head[x['frame_index']]['segment']==head[r['frame_index']]['segment']]);stamps=np.array([rows[j]['timestamp_ns'] for j in ids],np.int64)
        offsets=np.array(SHORT_OFFSETS);targets=int(r['timestamp_ns'])+np.rint(offsets*1e9).astype(np.int64);distance=np.abs(targets[:,None]-stamps[None]);near=distance.argmin(1);good=distance[np.arange(17),near]<=20_000_000
        win=ids[near];win[8]=i;good[8]=True;windows.append(win);dt.append((stamps[near]-r['timestamp_ns'])/1e9);valid.append(good)
    windows=torch.tensor(np.array(windows),device='cuda');dt=torch.tensor(np.array(dt),device='cuda',dtype=torch.float32);valid=torch.tensor(np.array(valid),device='cuda')
    results={};factors=[];native_parts=[];joints=[];states=[];deltas=[]
    with torch.inference_mode():
        for begin in range(0,N,8):
            end=min(N,begin+8);ids=windows[begin:end];v=valid[begin:end];s=state[ids];r=R[ids];t=T[ids]
            world=torch.einsum('btij,btj->bti',r,s[...,:3])+t;relative=torch.einsum('bji,btj->bti',r[:,8],world-world[:,8:9])/.3
            globalR=torch.einsum('bji,btjk,btkl->btil',r[:,8],r,exp_so3(s[...,3:6]));local=exp_so3(s[...,6:].reshape(len(ids),17,15,3))
            six=lambda x:x[...,:,:2].transpose(-1,-2).flatten(-2)
            tokens=torch.cat([torch.cat([relative,torch.zeros_like(relative)],-1)[:,:,None],six(globalR)[:,:,None],six(local)],2)
            b=dict(rgb=rgb[ids],coarse_tokens=tokens,valid=v,dt=dt[begin:end],scores=scores[ids],right_window=right[ids],
                state=state[begin:end],betas=beta[begin:end],right=right[begin:end])
            with torch.autocast('cuda',dtype=torch.bfloat16):result,delta,samples=predict(model,codec,b)
            alpha=torch.ones(len(ids),device='cuda')
            if a.protect:
                original=codec.from_state(b['state'],b['betas'],b['right']);lo=torch.zeros_like(alpha);hi=torch.ones_like(alpha)
                for _ in range(16):
                    mid=(lo+hi)/2;p=codec.apply_residual(b['state'],b['betas'],b['right'],delta*mid[:,None,None])['joints']
                    displacement=(p-original).norm(dim=-1)*1000
                    good=(displacement.square().mean(1).sqrt()<=5)&(torch.quantile(displacement,.95,dim=1)<=10)&(displacement.amax(1)<=30)
                    lo=torch.where(good,mid,lo);hi=torch.where(good,hi,mid)
                alpha=lo;delta=delta*alpha[:,None,None];result=codec.apply_residual(b['state'],b['betas'],b['right'],delta)
            native=codec.export_native(result);check=codec.decode_exported(native)
            assert float((check-result['joints']).abs().max())<1e-6
            for name,value in native.items():results.setdefault(name,[]).append(value.cpu().numpy())
            xyz=result['joints'].cpu().numpy();root=result['root_rotation'].cpu().numpy();local=result['local_rotation'].cpu().numpy()
            packed=np.concatenate([result['wrist'].cpu().numpy(),Rotation.from_matrix(root).as_rotvec(),Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(-1,45)],-1)
            states.append(packed.astype(np.float32));joints.append(xyz);factors.extend(alpha.cpu().tolist());deltas.append(delta.cpu().numpy())
    path=a.output/'native_parameters.npz'
    np.savez_compressed(path,state_camera51=np.concatenate(states),joints_camera21=np.concatenate(joints),right=coarse['right'],
        track_id=np.array([r['track_id'] for r in rows]),timestamp_ns=np.array([r['timestamp_ns'] for r in rows],np.int64),
        normalized_residual=np.concatenate(deltas),protection_alpha=np.array(factors),**{k:np.concatenate(v) for k,v in results.items()})
    manifest=dict(complete=True,GT_used=False,landmark_IK_used=False,native_MANO=True,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        parameter_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),observations=N,draws=4,DDIM_steps=10,
        aggregation='mean in MANO Lie residual tangent, no GT candidate scoring',wrist_translation_unchanged=True,shape_unchanged=True,
        protection='whole-pose parameter interpolation before FK, per-frame 5/10/30mm displacement limits' if a.protect else 'training loss only',
        export='actual MANO global_orient/hand_pose/betas/transl plus explicit WiLoR left reflection; MANO transl is not wrist XYZ',
        safe_default_admitted=False)
    (a.output/'seal.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))

if __name__=='__main__':main()
