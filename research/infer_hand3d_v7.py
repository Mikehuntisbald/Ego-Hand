"""Raw RGB + original WiLoR XYZ -> offline 3D candidates and validated policy output."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from hand3d_data_v7 import camera_pose,risk_features
from hand3d_temporal_v7 import WRIST
from hand3d_protected_v7 import ProtectedHand3D
from hand3d_risk_v7 import Risk3D
from evaluate_hand3d_bounded_v7 import apply_bounded,RUN
from infer_natural_reliability import encode_track
from infer_sampling_v6 import sampled_windows
from temporal_sampling_v6 import OFFSETS
from spatial_rgb_model import SpatialHead
from offline_rgb_encoder import crop_roi
import spatial_rgb_common as s

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--kind',choices=['regression','dit']);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    seal=RUN/'sealed';manifest=json.loads((seal/'manifest.json').read_text())
    for name,digest in manifest.items():assert hashlib.sha256((seal/name).read_bytes()).hexdigest()==digest,name
    selection=json.loads((seal/'selection.json').read_text());arm='rgb_'+a.kind if a.kind else selection.get('annotation_arm',selection['arm']);policy=json.loads((seal/'policies.json').read_text())[arm]['policy']
    ck=torch.load(seal/f'{arm}.pt',weights_only=False,map_location=a.device);model=ProtectedHand3D(ck['kind'],ck['use_rgb'],ck['width'],ck['depth']).to(a.device).eval();model.load_state_dict(ck['model'])
    risk_ck=torch.load(seal/'risk_all.pt',weights_only=False,map_location=a.device);risk=Risk3D(risk_ck['dim']).to(a.device).eval();risk.load_state_dict(risk_ck['model'])
    temperature=torch.tensor(json.loads((seal/'risk_calibration.json').read_text())['temperature'],device=a.device)
    probe=SpatialHead().to(a.device).eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',map_location=a.device,weights_only=False)['model'])
    projection=torch.load(s.common.ROOT/'experiments/natural_reliability_v4/sealed/risk_projection.pt',weights_only=False)
    full,_=s.common.load_model(a.device);encoder=full.backbone;del full
    source=json.loads(Path(a.input).read_text());assert source['image_size']==[1408,1408];tracks=[];locked_count=0;max_displacement=0.
    yy,xx=np.mgrid[:16,:12];canonical=np.stack([xx*16+37.5,yy*16+5.5],-1).reshape(192,2)
    for track in source['tracks']:
        # Explicit whitelist excludes any supplied GT and visibility fields.
        frames=[];native=[];available=[];confirmed=[];rotations=[];translations=[];rays_world=[]
        for f in track['frames']:
            xyz=np.asarray(f['xyz_camera_m'],np.float32);assert xyz.shape==(20,3)
            exists=np.array(f.get('available_3d',np.isfinite(xyz).all(-1)),bool);locks=np.array(f.get('confirmed_3d',[False]*20),bool)
            assert np.all(~locks|exists);assert np.isfinite(xyz[exists]).all();xyz=np.where(exists[:,None],xyz,0.)
            uv=np.asarray(f.get('xy_px',s.common.from_json(f['camera']).eye_to_window(xyz)),np.float32)
            valid2=np.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<1408).all(-1)&exists
            frames.append(dict(image=f['image'],camera=f['camera'],timestamp_s=f['timestamp_s'],box_xyxy=f['box_xyxy'],box_confidence=f['box_confidence'],
                clip=f.get('clip',0),xy_px=np.nan_to_num(uv).tolist(),available=valid2.tolist(),confirmed=[False]*20))
            native.append(xyz);available.append(exists);confirmed.append(locks);R,t=camera_pose(f['camera']);rotations.append(R);translations.append(t)
            roi=crop_roi(f['box_xyxy']);_,_,transform,focal,*_=s.prepare(dict(image=f['image'],camera=f['camera'],clip=f.get('clip',0)),roi,[[0,0,0,0]])
            rays=np.c_[(canonical-127.5)/focal,np.ones(192)].astype(np.float32)@transform.T;rays/=np.maximum(np.linalg.norm(rays,axis=-1,keepdims=True),1e-6);rays_world.append(rays@R.T)
        z=encode_track({'frames':frames},encoder,probe,projection,a.device);sampled,chosen=sampled_windows(z,OFFSETS['multiscale'],a.device)
        xyz=torch.tensor(np.stack(native),device=a.device);exists=torch.tensor(np.stack(available),device=a.device);locks=torch.tensor(np.stack(confirmed),device=a.device)
        R=torch.tensor(np.stack(rotations),device=a.device);t=torch.tensor(np.stack(translations),device=a.device);rw=torch.tensor(np.stack(rays_world),device=a.device)
        world=torch.einsum('njc,nkc->njk',xyz,R)+t[:,None];ids=torch.tensor([[i if i is not None else 0 for i in row] for row in chosen],device=a.device);slot_valid=sampled['rgb_valid']
        aligned=torch.einsum('ntjc,nck->ntjk',world[ids]-t[:,None,None],R);aligned_exists=exists[ids]&slot_valid[...,None];aligned=torch.where(aligned_exists[...,None],aligned,0.)
        rays=torch.einsum('ntsc,nck->ntsk',rw[ids],R);origin=torch.einsum('ntc,nck->ntk',t[ids]-t[:,None],R)*slot_valid[...,None]
        risk_rgb=z['risk_rgb'][ids.cpu().numpy()].to(a.device)*slot_valid[:,:,None,None]
        scores=torch.from_numpy(z['scores']).to(a.device)[ids]*slot_valid
        b=dict(xyz=aligned,available=aligned_exists,base=xyz,dt=sampled['dt'],xy=sampled['xy'],observed_2d=sampled['available'],
            rgb=sampled['rgb'],positions=sampled['positions'],roi=sampled['roi'],scores=scores,rays=rays,camera_origin=origin,rgb_valid=slot_valid,confirmed=locks,risk_rgb=risk_rgb)
        camera=[];relative=[]
        for start in range(0,len(frames),96):
            v={k:x[start:start+96] for k,x in b.items()};prob=(risk(risk_features(v))/temperature).sigmoid();camera.append(prob[:,:,0]);relative.append(prob[:,:,1])
        b['risk_camera']=torch.cat(camera);b['risk_relative']=torch.cat(relative);outputs=[]
        for start in range(0,len(frames),32):
            v={k:x[start:start+32] for k,x in b.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):candidate=model.predict(v)
            final,_=apply_bounded(candidate['xyz_camera_m'],v,policy)
            outputs.append(dict(xyz=final.cpu(),candidate=candidate['xyz_camera_m'].cpu(),std=candidate['std_m'].cpu()))
        out={k:torch.cat([r[k] for r in outputs]) for k in outputs[0]};rendered=[]
        for i,f in enumerate(track['frames']):
            joints=[]
            for j in range(20):
                value=out['xyz'][i,j].tolist();candidate=out['candidate'][i,j].tolist();is_locked=bool(confirmed[i][j]);provided=bool(available[i][j])
                unchanged=torch.equal(out['xyz'][i,j],xyz[i,j].cpu())
                if is_locked or unchanged:value=f['xyz_camera_m'][j]
                if is_locked:
                    assert torch.equal(out['candidate'][i,j],xyz[i,j].cpu());locked_count+=1;candidate=f['xyz_camera_m'][j]
                assert np.isfinite(value).all() and np.isfinite(candidate).all()
                delta=float(np.linalg.norm(np.asarray(value)-np.asarray(f['xyz_camera_m'][j]))*1000);max_displacement=max(max_displacement,delta)
                joints.append(dict(xyz_camera_m=value,candidate_xyz_camera_m=candidate,std_m=out['std'][i,j].tolist(),confirmed=is_locked,input_available=provided,
                    review_required=not is_locked,automatic_correction_applied=delta>1e-5,camera_error_risk=float(b['risk_camera'][i,j]),relative_error_risk=float(b['risk_relative'][i,j])))
            rendered.append(dict(image=f['image'],timestamp_s=f['timestamp_s'],context_frame_indices=chosen[i],joints=joints))
        tracks.append(dict(id=track.get('id'),frames=rendered))
    result=dict(mode='offline_3d_hand_completion',arm=arm,unit='meters',coordinate_frame='current camera',joint_order='HOT3D 20-joint mapping; wrist index5',policy=policy,
        automatic_policy_approved=selection['approved'],confirmed_checked=locked_count,max_camera_displacement_mm=max_displacement,tracks=tracks,
        note='Candidate XYZ is a true 3D model prediction. A failed automatic gate returns the supplied WiLoR XYZ while retaining the 3D candidate for review. Automatic confidence is error risk, not visibility or a guarantee.')
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=str(dest),frames=sum(len(t['frames']) for t in tracks),confirmed=locked_count,automatic_approved=selection['approved'])))

if __name__=='__main__':main()
