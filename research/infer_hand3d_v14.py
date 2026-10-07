"""Raw RGB + existing predicted hand tracks -> reviewed camera XYZ annotation."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from hand3d_v8_common import V7
from encode_hand3d_dense_v14 import encode,OFFSETS
from density_model_v13 import DensityTrajectoryHand3D,POLICY
from hand3d_data_v7 import risk_features
from hand3d_risk_v7 import Risk3D
from spatial_rgb_model import SpatialHead
from native_projection_policy_v11 import apply as conservative
from adaptive_projection_v14 import apply
import spatial_rgb_common as s

RUN=V7.parent/'adaptive_projection_v14/dit_dense'
MODEL=V7.parent/'native_density_v13/dit_dense/best.pt'
DATA=V7.parent/'aligned_density_v13'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def load_models(device):
    frozen=json.loads((RUN/'fourth_seal.json').read_text());approved=json.loads((RUN/'fourth_results.json').read_text())['passed']
    assert approved and sha(MODEL)==frozen['model_sha256']
    assert sha(DATA/'risk_dense/risk_all.pt')==frozen['risk_sha256']
    assert sha(DATA/'risk_dense/calibration.json')==frozen['temperature_sha256']
    for name,h in frozen['code_sha256'].items():assert sha(Path(__file__).resolve().parent/name)==h,name
    deployment=RUN/'deployment_seal.json'
    if deployment.exists():
        runtime=json.loads(deployment.read_text())
        for name,h in runtime['code_sha256'].items():assert sha(Path(__file__).resolve().parent/name)==h,name
        for path,h in runtime['asset_sha256'].items():assert sha(Path(path))==h,path
    model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(torch.load(MODEL,weights_only=False,map_location=device)['model'])
    ck=torch.load(DATA/'risk_dense/risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(ck['dim']).to(device).eval();risk.load_state_dict(ck['model'])
    temperature=torch.tensor(json.loads((DATA/'risk_dense/calibration.json').read_text())['temperature'],device=device)
    probe=SpatialHead().to(device).eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=device)['model'])
    projection=torch.load(V7.parent/'natural_reliability_v4/sealed/risk_projection.pt',weights_only=False)
    full,_=s.common.load_model(device);encoder=full.backbone;del full
    return model,risk,temperature,probe,projection,encoder,frozen

@torch.inference_mode()
def infer(source,device='cuda:2'):
    torch.set_num_threads(4);assert source['image_size']==[1408,1408]
    model,risk,temperature,probe,projection,encoder,frozen=load_models(device)
    tracks=[];checked=0;missing_hands=0;partial_lock_hands=0
    for track in source['tracks']:
        b,chosen=encode(track,encoder,probe,projection,device)
        prob=torch.cat([(risk(risk_features({k:v[start:start+64] for k,v in b.items()}))/temperature).sigmoid() for start in range(0,len(b['base']),64)])
        b['risk_camera']=prob[:,:,0];b['risk_relative']=prob[:,:,1];outputs=[]
        for start in range(0,len(b['base']),8):
            bb={k:v[start:start+8] for k,v in b.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):candidate=model.predict_rollout(bb,seed=202610114+start)
            raw=candidate['xyz_camera_m'].float();complete=bb['available'][:,8].all(-1);locked=bb['confirmed'].any(-1)
            safe=conservative(raw,bb['base'],POLICY,bb['confirmed'])
            adaptive=apply(raw,bb,frozen['policy'])
            # Accuracy of large-radius updates is validated on unlocked inputs.
            # Confirmed inputs use the conservative geometric protection policy.
            pred=torch.where(locked[:,None,None],safe,adaptive)
            pred=torch.where(complete[:,None,None],pred,bb['base'])
            assert torch.isfinite(raw).all() and torch.isfinite(pred).all()
            dc=(pred-bb['base']).norm(dim=-1);dr=((pred-pred[:,5:6])-(bb['base']-bb['base'][:,5:6])).norm(dim=-1)
            rc=torch.where(bb['risk_camera']>=frozen['policy']['camera_risk_threshold'],frozen['policy']['large_cap_m'],.0095)
            rr=torch.where(bb['risk_relative']>=frozen['policy']['relative_risk_threshold'],frozen['policy']['large_cap_m'],.0095)
            rc=torch.where(locked[:,None],.0095,rc);rr=torch.where(locked[:,None],.0095,rr)
            assert (dc<=rc+1e-6).all() and (dr<=rr+1e-6).all()
            outputs.append(dict(pred=pred.cpu(),raw=raw.cpu(),std=candidate['std_m'].float().cpu(),draws=candidate['draws'].float().cpu().transpose(0,1),complete=complete.cpu(),partial_lock=locked.cpu(),camera_radius=rc.cpu(),relative_radius=rr.cpu()))
        out={k:torch.cat([x[k] for x in outputs]) for k in outputs[0]};frames=[]
        for i,f in enumerate(track['frames']):
            complete=bool(out['complete'][i]);partial=bool(out['partial_lock'][i]);missing_hands+=not complete;partial_lock_hands+=partial
            mode='adaptive' if complete and not partial else 'conservative_confirmed' if complete else 'missing_xyz_review_only'
            joints=[]
            for j in range(20):
                exists=bool(b['available'][i,8,j]);lock=bool(b['confirmed'][i,j]);original=f['xyz_camera_m'][j]
                value=out['pred'][i,j].tolist() if exists else None;proposal=out['raw'][i,j].tolist()
                if exists and not complete:value=original
                if lock:
                    assert torch.equal(out['pred'][i,j],b['base'][i,j].cpu());value=proposal=original;checked+=1
                dc=float(np.linalg.norm(np.array(value)-np.array(original))*1000) if exists else None
                rawchange=float(np.linalg.norm(np.array(proposal)-np.array(original))*1000) if exists else None
                joints.append(dict(xyz_camera_m=value,candidate_xyz_camera_m=proposal,candidates_xyz_camera_m=[original]*4 if lock else out['draws'][i,:,j].tolist(),std_m=[0.,0.,0.] if lock else out['std'][i,j].tolist(),confirmed=lock,input_available=exists,review_required=not lock,needs_special_review=not lock and (not complete or dc>9.5 or rawchange>9.5),correction_applied=exists and dc>1e-5,camera_error_risk=float(prob[i,j,0]),relative_error_risk=float(prob[i,j,1]),camera_radius_mm=float(out['camera_radius'][i,j]*1000),relative_radius_mm=float(out['relative_radius'][i,j]*1000)))
            frames.append(dict(image=f['image'],timestamp_s=f['timestamp_s'],context_frame_indices=chosen[i],projection_mode=mode,automatic_projection_applied=complete,missing_3d_input_review_only=not complete,joints=joints))
        tracks.append(dict(id=track.get('id'),frames=frames))
    return dict(mode='offline_native3d_dit_v14',unit='meters',coordinate_frame='current camera',joint_order='HOT3D20; wrist5',output='20x3 XYZ; unavailable coordinates remain null with separate review hypotheses',whole_trajectory_internal='17x20x3',native_semantic_channels=1280,native_spatial_cells=192,offsets_seconds=(np.array(OFFSETS)/30).tolist(),automatic_policy_approved=True,policy=frozen['policy'],confirmed_checked=checked,missing_input_review_only_hands=missing_hands,conservative_confirmed_hands=partial_lock_hands,tracks=tracks,note='Fourth12unused clips passed fixed acceptance; encountered subjects/source sequences. Existing boxes/tracks required. All unconfirmed points require annotation review. Missing XYZ hypotheses have no accuracy validation. Raw draws and spread do not establish correctness or visibility. Severe occlusion remains imperfect.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:2');a=p.parse_args()
    result=infer(json.loads(Path(a.input).read_text()),a.device);dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(dest),confirmed=result['confirmed_checked'],missing_review_only=result['missing_input_review_only_hands'],conservative_confirmed_hands=result['conservative_confirmed_hands'])),flush=True)

if __name__=='__main__':main()
