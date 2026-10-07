"""Actual RGB/YOLO-track input -> offline stable parameterized 3D estimates.

Use --prepared for WiLoR XYZ already reconstructed by v16's upstream adapter.
Confirmed annotations are preserved separately; incompatible tracks receive
review-only model candidates, never a mixed unconstrained authoritative hand.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_v8_common import V7
from hand3d_data_v7 import camera_pose,risk_features
from hand3d_risk_v7 import Risk3D
from encode_hand3d_dense_v14 import encode
from semantic_parameter_model_v36 import SemanticParameterHand
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from joint_mano_model_v29 import six_to_rotation,rotation_to_six
from spatial_rgb_model import SpatialHead
from stability_trajectory_v42 import fit_stable,DEFAULT
import spatial_rgb_common as s


def load_models(device):
    run=V7.parent/'fitted_parameter_v39/regression'
    ck=torch.load(run/'best.pt',weights_only=False,map_location=device)
    assert ck['kind']=='regression' and 'coarse_seed' in ck['config']
    model=SemanticParameterHand('regression',device).to(device).eval();model.load_state_dict(ck['model'])
    riskpath=V7.parent/'side_data_v16/consensus/risk_dense'
    rc=torch.load(riskpath/'risk_all.pt',weights_only=False,map_location=device)
    risk=Risk3D(rc['dim']).to(device).eval();risk.load_state_dict(rc['model'])
    temp=torch.tensor(json.loads((riskpath/'calibration.json').read_text())['temperature'],device=device)
    probe=SpatialHead().to(device).eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=device)['model'])
    projection=torch.load(V7.parent/'natural_reliability_v4/sealed/risk_projection.pt',weights_only=False)
    full,_=s.common.load_model(device);encoder=full.backbone;del full
    return model,risk,temp,probe,projection,encoder


def fit_seeds(codec,xyz,right,steps=200):
    """Same observation-only IK objective as v38, with no label arguments."""
    with torch.no_grad():state=codec.coarse_states(xyz,right).flatten(1)
    n=len(state);device=xyz.device
    raw=nn.Parameter(torch.logit(((state[:,9:29]+1)/2).clamp(.001,.999)))
    R=nn.Parameter(state[:,3:9].clone());root=nn.Parameter(state[:,:3].clone()*.1)
    sh=nn.Parameter(torch.zeros(n,5,device=device));group=torch.arange(n,device=device);sign=1-2*right.float()
    opt=torch.optim.Adam([dict(params=[raw],lr=.04),dict(params=[R],lr=.01),dict(params=[root],lr=.001),dict(params=[sh],lr=.01)])
    for _ in range(steps):
        beta=4*sh.tanh();pred=codec(raw,R,root,beta,group,sign)
        loss=F.smooth_l1_loss((pred-xyz)/.01,torch.zeros_like(pred),beta=.5)+.0001*beta.square().mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    with torch.no_grad():
        state[:,:3]=root/.1;state[:,3:9]=rotation_to_six(six_to_rotation(R))
        lo,hi=codec.joint_limits[:20].unbind(-1)
        state[:,9:29]=2*(codec.angles(raw)[:,:20]-lo)/(hi-lo)-1;state[:,29:34]=4*sh.tanh()
    return state.reshape(n,21,3).detach()


def add_parameter_windows(b,chosen,fitted,right,R,T):
    n=len(chosen);device=R.device
    ids=torch.tensor([[k if k is not None else 0 for k in row] for row in chosen],device=device)
    state=fitted[ids].clone();flat=state.flatten(2)
    local_rotation=six_to_rotation(flat[:,:,3:9])
    flat[:,:,3:9]=rotation_to_six(R.transpose(-1,-2)[:,None]@R[ids]@local_rotation)
    world_root=torch.einsum('ntj,ntkj->ntk',flat[:,:,:3]*.1,R[ids])+T[ids]
    flat[:,:,:3]=torch.einsum('ntj,njk->ntk',world_root-T[:,None],R)/.1
    state*=b['rgb_valid'][:,:,None,None]
    b['kinematic_coarse']=state;b['predicted_right']=right


def complete(source,device='cuda:0',prepared=False,steps=350):
    torch.set_num_threads(4)
    if not prepared:
        from complete_hand_tracks_v16 import reconstruct
        source=reconstruct(source,device)
    model,risk,temp,probe,projection,encoder=load_models(device)
    decoder=ParameterTrajectoryDecoder(device);outputs=[];checks=[]
    for ti,track in enumerate(source['tracks']):
        # Keep arrays alive only for one track, including its 17 spatial windows.
        b,chosen=encode(track,encoder,probe,projection,device)
        R=[];T=[];rows=[];camera_params=[]
        for i,f in enumerate(track['frames']):
            rotation,translation=camera_pose(f['camera']);R.append(rotation);T.append(translation)
            cam=s.common.from_json(f['camera']);camera_params.append(list(cam.f)+list(cam.c)+list(cam.distort))
            rows.append(dict(sequence=f'runtime{ti}',clip=f.get('clip',0),track_id=ti,
                             timestamp_ns=int(round(f['timestamp_s']*1e9)),frame=i))
        R=torch.tensor(np.asarray(R),device=device);T=torch.tensor(np.asarray(T),device=device)
        right=torch.tensor([f['predicted_right'] for f in track['frames']],device=device)
        fitted=fit_seeds(decoder,b['base'].detach(),right)
        add_parameter_windows(b,chosen,fitted,right,R,T)
        with torch.inference_mode():
            probability=torch.cat([(risk(risk_features({k:v[begin:begin+64] for k,v in b.items()}))/temp).sigmoid() for begin in range(0,len(rows),64)])
            b['risk_camera']=probability[:,:,0];b['risk_relative']=probability[:,:,1]
            values=[];states=[];sides=[];heat=[]
            for begin in range(0,len(rows),8):
                bb={k:v[begin:begin+8] for k,v in b.items()}
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model.generate(bb,202610131+begin)
                values.append(p['xyz'][:,8].float());states.append(p['state'][:,8].float())
                sides.append(p['right']);heat.append(p['encoded'][3]['xy'][:,8].float())
            obs=dict(xyz=torch.cat(values),parameter_state=torch.cat(states),right=torch.cat(sides))
        cache=dict(base=b['base'].detach(),rotation=R,translation=T,risk=probability,
                   heat_xy=torch.cat(heat),roi=b['roi'][:,8],camera_params=torch.tensor(camera_params,device=device),confirmed=b['confirmed'])
        result=fit_stable(decoder,cache,obs,rows,dict(DEFAULT,steps=steps))
        blocked=bool(result['manual_anchor_conflicts'].any());frames=[]
        for i,f in enumerate(track['frames']):
            locks=f.get('confirmed_3d',[False]*20)
            frame=dict(image=f['image'],timestamp_s=f['timestamp_s'],
                       xyz_camera_m=None if blocked else result['prediction'][i].cpu().tolist(),
                       candidate_xyz_camera_m=result['prediction'][i].cpu().tolist(),
                       annotation_anchors=[f['xyz_camera_m'][j] if locks[j] else None for j in range(20)],
                       confirmed_3d=locks,review_required=True,manual_anchor_conflict=bool(result['manual_anchor_conflicts'][i].any()),
                       predicted_right=bool(result['parameters']['sign'][result['layout']['source_map'][i]]<0),
                       parameter_frame_index=int(result['layout']['source_map'][i]),context_frame_indices=chosen[i])
            frames.append(frame)
        p=result['parameters'];angles=decoder.angles(p['local'])[:,:20];rotation=six_to_rotation(p['global_six'])
        dense=[];layout=result['layout']
        for j in range(len(result['world'])):
            dense.append(dict(timestamp_s=float(layout['times'][j]),interpolated=bool(layout['left'][j]!=layout['right'][j]),
                xyz_world_m=result['world'][j].cpu().tolist(),root_world_m=p['root'][j].cpu().tolist(),
                rotation_world_matrix=rotation[j].cpu().tolist(),joint_angles_rad=angles[j].cpu().tolist(),
                shape_coefficients=p['beta'][p['group'][j]].cpu().tolist(),predicted_right=bool(p['sign'][j]<0)))
        checks.append(result['check']);outputs.append(dict(id=track.get('id'),frames=frames,dense_world_trajectory=dense,
                segment_frame_indices=layout['segments'],restoration=result['restoration'],constraints=result['check'],
                manual_anchor_blocked=blocked,status='review_only_manual_anchor_conflict' if blocked else 'stable_estimate'))
        print(json.dumps(dict(track=track.get('id'),frames=len(rows),constraints_passed=result['check']['passed'],manual_conflict=blocked)),flush=True)
    return dict(mode='offline_stability_first_v42',unit='meters',coordinate_frame='current camera for observed frames, world for densified trajectory',
        output='HOT3D20 wrist5, decoded from shared-shape bounded hand parameters',tracks=outputs,
        config=dict(DEFAULT,steps=steps),constraint_checks_passed=all(x['passed'] for x in checks),
        raw_xyz_fallback_frames=0,accuracy_certified=False,rgb_conditioned=True,noncausal=True,
        note='Plausible stable annotation estimates; occluded poses and wrong depth remain uncertain. Dense virtual times are estimates, not actual captured frames. No collision guarantee, no reassociation across IDs or gaps>0.55s. Incompatible confirmed annotations preserved separately; candidate track review-only.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda:0');ap.add_argument('--prepared',action='store_true');ap.add_argument('--steps',type=int,default=350)
    a=ap.parse_args();source=json.loads(Path(a.input).read_text());result=complete(source,a.device,a.prepared,a.steps)
    out=Path(a.output);out.parent.mkdir(exist_ok=True,parents=True);out.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(out),constraints_passed=result['constraint_checks_passed'])),flush=True)


if __name__=='__main__':main()
