"""Natural RGB/XYZ to3D trajectory candidates and validated coherent correction."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,torch
from infer_hand3d_v8 import encode
from hand3d_trajectory_v9 import TrajectoryHand3D
from proposal_critic_v9 import ProposalCritic,features
from calibrate_proposal_critic_v9 import choose
from hand3d_data_v7 import risk_features
from hand3d_risk_v7 import Risk3D
from spatial_rgb_model import SpatialHead
from bounded_policy_v8 import apply
import spatial_rgb_common as s

@torch.inference_mode()
def main(version='v9'):
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    experiments=s.common.ROOT/'experiments';run=experiments/('native_critic_v10' if version=='v10' else 'proposal_critic_v9');proposal=experiments/('offline_hand3d_v10_rollout/rgb_dit/best.pt' if version=='v10' else 'offline_hand3d_v9_rollout/rgb_dit/best.pt');seal=json.loads((run/'fresh_evaluation_seal.json').read_text());approved=json.loads((run/'fresh_results.json').read_text())['passed']
    assert hashlib.sha256(proposal.read_bytes()).hexdigest()==seal['proposal_sha256'];assert hashlib.sha256((run/'critic.pt').read_bytes()).hexdigest()==seal['critic_sha256']
    ck=torch.load(proposal,weights_only=False,map_location=a.device)
    if version=='v10':
        from hand3d_native_v10 import NativeTrajectoryHand3D
        model=NativeTrajectoryHand3D('dit',True).to(a.device).eval()
    else:model=TrajectoryHand3D('dit').to(a.device).eval()
    model.load_state_dict(ck['model']);cck=torch.load(run/'critic.pt',weights_only=False,map_location=a.device);critic=ProposalCritic(cck['dim']).to(a.device).eval();critic.load_state_dict(cck['model'])
    rck=torch.load(experiments/'offline_hand3d_v7/risk_all.pt',weights_only=False,map_location=a.device);risk=Risk3D(rck['dim']).to(a.device).eval();risk.load_state_dict(rck['model']);temperature=torch.tensor(json.loads((experiments/'offline_hand3d_v7/risk_calibration.json').read_text())['temperature'],device=a.device)
    probe=SpatialHead().to(a.device).eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=a.device)['model']);projection=torch.load(experiments/'natural_reliability_v4/sealed/risk_projection.pt',weights_only=False);full,_=s.common.load_model(a.device);encoder=full.backbone;del full
    obj=json.loads(Path(a.input).read_text());assert obj['image_size']==[1408,1408];tracks=[];locks_checked=0
    for track in obj['tracks']:
        b,chosen=encode(track,encoder,probe,projection,a.device,return_native=version=='v10');pr=torch.cat([(risk(risk_features({k:v[start:start+64] for k,v in b.items()}))/temperature).sigmoid() for start in range(0,len(b['base']),64)]);b['risk_camera']=pr[:,:,0];b['risk_relative']=pr[:,:,1];params=[]
        for f in track['frames']:
            cam=s.common.from_json(f['camera']);params.append(list(cam.f)+list(cam.c)+list(cam.distort))
        params=torch.tensor(params,device=a.device);output=[]
        for start in range(0,len(b['base']),8):
            bb={k:v[start:start+8] for k,v in b.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(bb,seed=(202610103 if version=='v10' else 202610093)+start)
            raw=p['xyz_camera_m'].float();std=p['std_m'].float();scores=torch.stack([critic(features(bb,raw,std,params[start:start+8],strength)) for strength in [.25,.5,.75,1.]])
            pred,accepted,alpha=choose(raw,bb['base'],scores,seal['policy'])
            # Partial confirmed locks change the candidate distribution. Preserve
            # them exactly and retain bounded behavior on that engineering path.
            has_locks=bb['confirmed'].any(-1);fallback=apply(raw,bb['base'],dict(cap_m=.0095,strength=.5),bb['confirmed']);pred=torch.where(has_locks[:,None,None],fallback,pred);accepted&=~has_locks
            if not approved:pred=bb['base'];accepted.zero_()
            output.append(dict(pred=pred.cpu(),raw=raw.cpu(),std=std.cpu(),draws=p['draws'].float().cpu().transpose(0,1),accepted=accepted.cpu(),alpha=alpha.cpu()))
        out={k:torch.cat([p[k] for p in output]) for k in output[0]};frames=[]
        for i,f in enumerate(track['frames']):
            joints=[]
            for j in range(20):
                locked=bool(b['confirmed'][i,j]);original=f['xyz_camera_m'][j];value=out['pred'][i,j].tolist();candidate=out['raw'][i,j].tolist()
                if locked:assert torch.equal(out['pred'][i,j],b['base'][i,j].cpu());value=candidate=original;locks_checked+=1
                distance=float(np.linalg.norm(np.asarray(candidate)-np.asarray(original))*1000)
                joints.append(dict(xyz_camera_m=value,candidate_xyz_camera_m=candidate,candidates_xyz_camera_m=[original]*4 if locked else out['draws'][i,:,j].tolist(),std_m=[0.,0.,0.] if locked else out['std'][i,j].tolist(),confirmed=locked,review_required=not locked,input_available=bool(b['available'][i,8,j]),correction_applied=float(np.linalg.norm(np.asarray(value)-np.asarray(original)))>1e-8,needs_special_review=(not locked and distance>9.5 and not bool(out['accepted'][i])),camera_error_risk=float(b['risk_camera'][i,j]),relative_error_risk=float(b['risk_relative'][i,j])))
            frames.append(dict(image=f['image'],timestamp_s=f['timestamp_s'],context_frame_indices=chosen[i],large_hand_correction_accepted=bool(out['accepted'][i]),large_correction_strength=float(out['alpha'][i]) if out['accepted'][i] else None,joints=joints))
        tracks.append(dict(id=track.get('id'),frames=frames))
    result=dict(mode=f'offline_3d_trajectory_dit_{version}',output='20x3 current-camera XYZ meters',whole_trajectory_internal='17x20x3',native_semantic_channels=1280 if version=='v10' else 128,automatic_policy_approved=approved,policy=seal['policy'],confirmed_checked=locks_checked,tracks=tracks,note='Whole-hand proposal judgment plus bounded fallback passed a frozen12clip validation. All unconfirmed points need annotation review; full candidate and individual draws are preserved. Severe occlusion remains unresolved. Pretrained/shared encoder overlap is not excluded.')
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=str(dest),automatic_approved=approved,confirmed=locks_checked)),flush=True)

if __name__=='__main__':main()
