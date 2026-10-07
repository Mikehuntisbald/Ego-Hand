"""Original RGB and WiLoR20x3 -> bounded 3D correction plus review candidates."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from hand3d_rollout_v8 import RolloutHand3D
from hand3d_data_v7 import camera_pose,risk_features
from hand3d_risk_v7 import Risk3D
from hand3d_temporal_v7 import WRIST
from bounded_policy_v8 import apply
from infer_natural_reliability import encode_track
from infer_sampling_v6 import sampled_windows
from temporal_sampling_v6 import OFFSETS
from spatial_rgb_model import SpatialHead
from offline_rgb_encoder import crop_roi
import spatial_rgb_common as s

def encode(track,encoder,probe,projection,device,return_native=False):
    frames=[];native=[];available=[];confirmed=[];rotations=[];translations=[];rays_world=[]
    yy,xx=np.mgrid[:16,:12];canonical=np.stack([xx*16+37.5,yy*16+5.5],-1).reshape(192,2)
    for f in track['frames']:
        xyz=np.asarray(f['xyz_camera_m'],np.float32);assert xyz.shape==(20,3)
        exists=np.asarray(f.get('available_3d',np.isfinite(xyz).all(-1)),bool);locks=np.asarray(f.get('confirmed_3d',[False]*20),bool)
        assert np.all(~locks|exists) and np.isfinite(xyz[exists]).all();xyz=np.where(exists[:,None],xyz,0.)
        uv=np.asarray(f.get('xy_px',s.common.from_json(f['camera']).eye_to_window(xyz)),np.float32)
        valid2=np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<1408).all(-1)&exists
        frames.append(dict(image=f['image'],camera=f['camera'],timestamp_s=f['timestamp_s'],box_xyxy=f['box_xyxy'],box_confidence=f['box_confidence'],clip=f.get('clip',0),xy_px=np.nan_to_num(uv).tolist(),available=valid2.tolist(),confirmed=[False]*20))
        native.append(xyz);available.append(exists);confirmed.append(locks);R,t=camera_pose(f['camera']);rotations.append(R);translations.append(t)
        roi=crop_roi(f['box_xyxy']);_,_,transform,focal,*_=s.prepare(dict(image=f['image'],camera=f['camera'],clip=f.get('clip',0)),roi,[[0,0,0,0]])
        rays=np.c_[(canonical-127.5)/focal,np.ones(192)].astype(np.float32)@transform.T;rays/=np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-6);rays_world.append(rays@R.T)
    z=encode_track({'frames':frames},encoder,probe,projection,device,return_native=return_native);sampled,chosen=sampled_windows(z,OFFSETS['multiscale'],device)
    xyz=torch.tensor(np.stack(native),device=device);exists=torch.tensor(np.stack(available),device=device);locks=torch.tensor(np.stack(confirmed),device=device);R=torch.tensor(np.stack(rotations),device=device);t=torch.tensor(np.stack(translations),device=device);rw=torch.tensor(np.stack(rays_world),device=device)
    world=torch.einsum('njc,nkc->njk',xyz,R)+t[:,None];ids=torch.tensor([[i if i is not None else 0 for i in row] for row in chosen],device=device);slot_valid=sampled['rgb_valid']
    aligned=torch.einsum('ntjc,nck->ntjk',world[ids]-t[:,None,None],R);aligned_exists=exists[ids]&slot_valid[...,None];aligned=torch.where(aligned_exists[...,None],aligned,0.)
    rays=torch.einsum('ntsc,nck->ntsk',rw[ids],R);origin=torch.einsum('ntc,nck->ntk',t[ids]-t[:,None],R)*slot_valid[...,None]
    b=dict(xyz=aligned,available=aligned_exists,base=xyz,dt=sampled['dt'],xy=sampled['xy'],observed_2d=sampled['available'],rgb=sampled['rgb'],positions=sampled['positions'],roi=sampled['roi'],scores=torch.from_numpy(z['scores']).to(device)[ids]*slot_valid,rays=rays,camera_origin=origin,rgb_valid=slot_valid,confirmed=locks,risk_rgb=z['risk_rgb'][ids.cpu().numpy()].to(device)*slot_valid[:,:,None,None])
    if return_native:b['rgb_native']=z['native'][ids.cpu().numpy()].to(device)*slot_valid[:,:,None,None]
    return b,chosen

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--run',default='/mnt/why/HOT3D/experiments/offline_hand3d_v8');p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    run=Path(a.run);seal=json.loads((run/'fresh_evaluation_seal.json').read_text());approved=json.loads((run/'fresh_results.json').read_text())['passed'];folder=seal['folder'];checkpoint=run/folder/'best.pt'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==seal['checkpoint_sha256'];assert folder=='dit_rollout','This raw adapter is sealed for the selected frozen-native RGB model.'
    ck=torch.load(checkpoint,weights_only=False,map_location=a.device);model=RolloutHand3D(ck['kind'],True).to(a.device).eval();model.load_state_dict(ck['model'])
    v7=run.parent/'offline_hand3d_v7';risk_ck=torch.load(v7/'risk_all.pt',weights_only=False,map_location=a.device);risk=Risk3D(risk_ck['dim']).to(a.device).eval();risk.load_state_dict(risk_ck['model']);temperature=torch.tensor(json.loads((v7/'risk_calibration.json').read_text())['temperature'],device=a.device)
    probe=SpatialHead().to(a.device).eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=a.device)['model']);projection=torch.load(run.parent/'natural_reliability_v4/sealed/risk_projection.pt',weights_only=False)
    full,_=s.common.load_model(a.device);encoder=full.backbone;del full
    source=json.loads(Path(a.input).read_text());assert source['image_size']==[1408,1408];tracks=[];checked=0;max_camera=max_relative=0.
    for track in source['tracks']:
        b,chosen=encode(track,encoder,probe,projection,a.device);prob=[]
        for start in range(0,len(b['base']),64):prob.append((risk(risk_features({k:v[start:start+64] for k,v in b.items()}))/temperature).sigmoid())
        prob=torch.cat(prob);b['risk_camera']=prob[:,:,0];b['risk_relative']=prob[:,:,1];out=[]
        for start in range(0,len(b['base']),16):
            v={k:x[start:start+16] for k,x in b.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):candidate=model.predict(v,seed=202610081+start)
            raw=torch.where(v['confirmed'][...,None],v['base'],candidate['raw_xyz_camera_m'].float());final=apply(candidate[seal['policy']['source']].float(),v['base'],seal['policy'],v['confirmed']) if approved else v['base']
            delta=final-v['base'];relative=delta-delta[:,WRIST:WRIST+1];max_camera=max(max_camera,float(delta.norm(dim=-1).max()*1000));max_relative=max(max_relative,float(relative.norm(dim=-1).max()*1000));assert max_camera<=9.501 and max_relative<=9.501
            out.append(dict(xyz=final.cpu(),candidate=raw.cpu(),std=candidate['std_m'].float().cpu(),draws=candidate['draws'].float().cpu().transpose(0,1),trust=candidate['trust'].float().cpu()))
        result={k:torch.cat([r[k] for r in out]) for k in out[0]};frames=[]
        for i,f in enumerate(track['frames']):
            joints=[]
            for j in range(20):
                locked=bool(b['confirmed'][i,j]);original=f['xyz_camera_m'][j];final=result['xyz'][i,j].tolist();raw=result['candidate'][i,j].tolist();delta=float(np.linalg.norm(np.asarray(final)-np.asarray(original))*1000)
                if locked:assert torch.equal(result['xyz'][i,j],b['base'][i,j].cpu());final=raw=original;checked+=1
                large=float(np.linalg.norm(np.asarray(raw)-np.asarray(original))*1000)>9.5
                joints.append(dict(xyz_camera_m=final,candidate_xyz_camera_m=raw,candidates_xyz_camera_m=[original]*4 if locked else result['draws'][i,:,j].tolist(),std_m=[0.,0.,0.] if locked else result['std'][i,j].tolist(),confirmed=locked,input_available=bool(b['available'][i,8,j]),correction_applied=delta>1e-5,review_required=not locked,needs_special_review=large and not locked,review_reason='candidate_requires_large_change' if large and not locked else 'unconfirmed_prediction' if not locked else 'confirmed_input',camera_error_risk=float(b['risk_camera'][i,j]),relative_error_risk=float(b['risk_relative'][i,j])))
            frames.append(dict(image=f['image'],timestamp_s=f['timestamp_s'],context_frame_indices=chosen[i],proposal_trust_score=float(result['trust'][i]),joints=joints))
        tracks.append(dict(id=track.get('id'),frames=frames))
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);obj=dict(mode='offline_3d_rgb_dit_v8',unit='meters',coordinate_frame='current camera',joint_order='HOT3D20; wrist5',automatic_policy_approved=approved,policy=seal['policy'],confirmed_checked=checked,max_camera_displacement_mm=max_camera,max_relative_displacement_mm=max_relative,tracks=tracks,note='Automatic bounded correction passed the frozen12clip benchmark. All unconfirmed points require annotation review. Full candidate and individual draws are retained; scores do not establish visibility or correctness. Existing hand box/track is required.')
    dest.write_text(json.dumps(obj,indent=2));print(json.dumps(dict(output=str(dest),automatic_approved=approved,confirmed=checked,max_camera_mm=max_camera,max_relative_mm=max_relative)),flush=True)

if __name__=='__main__':main()
