"""Original RGB + camera XYZ -> native3D DiT annotation proposals.

Existing boxes/tracks are required. GT/visibility/shape fields are ignored.
Only fully available3D inputs use the validated automatic projection policy.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from infer_hand3d_v8 import encode
from hand3d_native_v10 import NativeTrajectoryHand3D
from hand3d_data_v7 import risk_features
from hand3d_risk_v7 import Risk3D
from spatial_rgb_model import SpatialHead
from native_projection_policy_v11 import apply
import spatial_rgb_common as s

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:2');args=p.parse_args();torch.set_num_threads(4)
    experiments=s.common.ROOT/'experiments';run=experiments/'native_projection_v11';model_path=experiments/'offline_hand3d_v10_rollout/rgb_dit/best.pt';frozen=json.loads((run/'fresh_evaluation_seal.json').read_text());approved=json.loads((run/'fresh_results.json').read_text())['passed']
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==frozen['model_sha256']['dit']
    for name in ['hand3d_native_v10.py','hand3d_trajectory_v9.py','native_projection_policy_v11.py','hand3d_data_v7.py']:
        assert hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest()==frozen['code_sha256'][name]
    ck=torch.load(model_path,weights_only=False,map_location=args.device);model=NativeTrajectoryHand3D('dit',True).to(args.device).eval();model.load_state_dict(ck['model'])
    rck=torch.load(experiments/'offline_hand3d_v7/risk_all.pt',weights_only=False,map_location=args.device);risk=Risk3D(rck['dim']).to(args.device).eval();risk.load_state_dict(rck['model']);temperature=torch.tensor(json.loads((experiments/'offline_hand3d_v7/risk_calibration.json').read_text())['temperature'],device=args.device)
    probe=SpatialHead().to(args.device).eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=args.device)['model']);projection=torch.load(experiments/'natural_reliability_v4/sealed/risk_projection.pt',weights_only=False)
    full,_=s.common.load_model(args.device);encoder=full.backbone;del full
    obj=json.loads(Path(args.input).read_text());assert obj['image_size']==[1408,1408];tracks=[];locks_checked=0;max_camera=max_relative=0.;review_only_hands=0
    for track in obj['tracks']:
        b,chosen=encode(track,encoder,probe,projection,args.device,return_native=True)
        pr=torch.cat([(risk(risk_features({k:v[start:start+64] for k,v in b.items()}))/temperature).sigmoid() for start in range(0,len(b['base']),64)]);b['risk_camera']=pr[:,:,0];b['risk_relative']=pr[:,:,1];output=[]
        for start in range(0,len(b['base']),8):
            bb={k:v[start:start+8] for k,v in b.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(bb,seed=202610103+start)
            raw=p['xyz_camera_m'].float();bounded=apply(raw,bb['base'],frozen['policy'],bb['confirmed']);complete=bb['available'][:,8].all(-1);review_only_hands+=int((~complete).sum())
            # An absent XYZ is a placeholder, so a9.5mm update around zero is
            # meaningless. Such hands remain review-only, preserving all known
            # input points and exposing missing-point hypotheses separately.
            pred=torch.where(complete[:,None,None],bounded,bb['base']) if approved else bb['base']
            dc=(pred-bb['base']).norm(dim=-1).max();dr=((pred-pred[:,5:6])-(bb['base']-bb['base'][:,5:6])).norm(dim=-1).max();max_camera=max(max_camera,float(dc*1000));max_relative=max(max_relative,float(dr*1000))
            assert max_camera<=9.501 and max_relative<=9.501
            output.append(dict(pred=pred.cpu(),raw=raw.cpu(),std=p['std_m'].float().cpu(),draws=p['draws'].float().cpu().transpose(0,1),complete=complete.cpu()))
        out={k:torch.cat([p[k] for p in output]) for k in output[0]};frames=[]
        for i,f in enumerate(track['frames']):
            joints=[]
            for j in range(20):
                locked=bool(b['confirmed'][i,j]);available=bool(b['available'][i,8,j]);original=f['xyz_camera_m'][j];value=out['pred'][i,j].tolist() if available else None;candidate=out['raw'][i,j].tolist()
                if available and not bool(out['complete'][i]):value=original
                if locked:assert torch.equal(out['pred'][i,j],b['base'][i,j].cpu());value=candidate=original;locks_checked+=1
                change=float(np.linalg.norm(np.asarray(candidate)-np.asarray(original))*1000) if available else None
                joints.append(dict(xyz_camera_m=value,candidate_xyz_camera_m=candidate,candidates_xyz_camera_m=[original]*4 if locked else out['draws'][i,:,j].tolist(),std_m=[0.,0.,0.] if locked else out['std'][i,j].tolist(),confirmed=locked,input_available=available,review_required=not locked,needs_special_review=not locked and (not available or change>9.5),correction_applied=bool(available and np.linalg.norm(np.asarray(value)-np.asarray(original))>1e-8),camera_error_risk=float(b['risk_camera'][i,j]),relative_error_risk=float(b['risk_relative'][i,j])))
            frames.append(dict(image=f['image'],timestamp_s=f['timestamp_s'],context_frame_indices=chosen[i],automatic_projection_applied=approved and bool(out['complete'][i]),missing_3d_input_review_only=not bool(out['complete'][i]),joints=joints))
        tracks.append(dict(id=track.get('id'),frames=frames))
    result=dict(mode='offline_native3d_dit_v11',unit='meters',coordinate_frame='current camera',output='HOT3D20 XYZ; missing inputs keep null and expose review candidate',whole_trajectory_internal='17x20x3',native_semantic_channels=1280,native_spatial_cells=192,automatic_policy_approved=approved,policy=frozen['policy'],confirmed_checked=locks_checked,missing_input_review_only_hands=review_only_hands,max_camera_displacement_mm=max_camera,max_relative_displacement_mm=max_relative,tracks=tracks,note='Validated on third12unused clips with existing boxes/tracks. All unconfirmed predictions require annotation review. Raw proposals and draws are hypotheses; sampling spread is not calibrated correctness. Severe occlusion remains unresolved. Reused subjects/source sequences; pretrained overlap not excluded.')
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=str(dest),automatic_approved=approved,confirmed=locks_checked,missing_review_only=review_only_hands)),flush=True)

if __name__=='__main__':main()
