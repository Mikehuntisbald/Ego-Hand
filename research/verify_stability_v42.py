"""Meaningful numeric, observation-whitelist, and real-RGB runtime checks."""
import argparse,collections,json,sys,subprocess
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from hand3d_v8_common import V7,save
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from evaluate_joint_kinematic_v30 import subset
from stability_trajectory_v42 import dense_layout,exp_rotation,fit_stable,saved_check


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='stability_first_v42_sidefix');args=ap.parse_args()
    torch.set_num_threads(4);device='cuda:0';root=V7.parent;out=root/args.run
    source=root/'joint_mano_v28';allrows=json.loads((source/'rows.json').read_text());ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select']);rows=[allrows[i] for i in ids]
    full=subset(torch.load(source/'candidates.pt',weights_only=False,mmap=True),ids,len(allrows),device)
    obs={k:v.to(device) for k,v in torch.load(root/'fitted_parameter_trajectory_v39/observations.pt',weights_only=False).items()}
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    ix=next(v for v in groups.values() if len(v)>25)[:16];ix=sorted(ix,key=lambda i:rows[i]['timestamp_ns']);indices=torch.tensor(ix,device=device)
    cc=subset(full,indices,len(rows),device);oo=subset(obs,indices,len(rows),device);rr=[rows[i] for i in ix]
    decoder=ParameterTrajectoryDecoder(device)
    # Pure numeric SO3 check near zero, pi, and beyond a full turn.
    rv=torch.tensor([[0.,0.,0.],[1e-7,0.,0.],[0.,np.pi-1e-6,0.],[0.,0.,7.]],device=device)
    reference=torch.tensor(Rotation.from_rotvec(rv.cpu().numpy()).as_matrix(),device=device)
    assert torch.allclose(exp_rotation(rv).double(),reference,atol=5e-7,rtol=0)
    one=dense_layout(rr[:1]);assert len(one['source_map'])==1 and len(one['segments'])==1
    gap=[dict(r) for r in rr[:3]];gap[1]['timestamp_ns']=gap[0]['timestamp_ns']+100_000_000;gap[2]['timestamp_ns']=gap[1]['timestamp_ns']+1_000_000_000
    layout=dense_layout(gap);assert len(layout['segments'])==2 and len(layout['times'])==5
    duplicate=[dict(rr[0]),dict(rr[0])]
    try:dense_layout(duplicate);raise AssertionError('Duplicate time accepted')
    except ValueError:pass
    a=fit_stable(decoder,cc,oo,rr,dict(steps=12));poison=dict(cc,gt=torch.full_like(cc['base'],float('nan')),valid=torch.zeros(len(rr),20,device=device,dtype=torch.bool),gt_handedness=1-oo['right'])
    b=fit_stable(decoder,poison,oo,rr,dict(steps=12))
    assert torch.equal(a['prediction'],b['prediction']);assert a['check']['passed']
    # A real track's inconsistent chirality is handled before parameter
    # interpolation. This regression test does not use accuracy labels.
    flip_indices=[i for i,r in enumerate(rows) if r['clip']==2114 and r['track_id']==81]
    fi=torch.tensor(flip_indices,device=device);fc=subset(full,fi,len(rows),device);fo=subset(obs,fi,len(rows),device);fr=[rows[i] for i in flip_indices]
    fixed=fit_stable(decoder,fc,fo,fr,dict(steps=5));assert fixed['side_parameter_outlier_frames']==8 and fixed['check']['passed']
    oneout=fit_stable(decoder,subset(cc,torch.tensor([0],device=device),len(rr),device),subset(oo,torch.tensor([0],device=device),len(rr),device),rr[:1],dict(steps=3))
    assert oneout['check']['passed']
    locked=dict(cc,confirmed=torch.zeros(len(rr),20,device=device,dtype=torch.bool));locked['confirmed'][0,0]=True
    conflict=fit_stable(decoder,locked,oo,rr,dict(steps=3));assert conflict['manual_anchor_conflicts'][0,0]
    # Frozen main result redecoded from disk, original frame camera transform.
    frozen=torch.load(out/'results.pt',weights_only=False);pars={k:v.to(device) for k,v in frozen['parameters'].items()}
    world,check=saved_check(decoder,pars,frozen['layout'],frozen['config']);assert check['passed']
    pred=torch.einsum('njc,nck->njk',world[torch.tensor(frozen['layout']['source_map'],device=device)]-full['translation'][:,None],full['rotation'])
    assert torch.allclose(pred.cpu(),frozen['prediction'],atol=1e-7,rtol=0)
    recordcache={};frames=[]
    for i in ix:
        r=rows[i];sp,seq,clip=Path(r['image']).relative_to('/mnt/why/HOT3D/export/images').parts[:3]
        path=root.parent/'export/annotations'/sp/seq/(clip+'.jsonl')
        if path not in recordcache:recordcache[path]=[json.loads(z) for z in path.read_text().splitlines()]
        camera=recordcache[path][r['frame']]['camera']
        frames.append(dict(image=r['image'],camera=camera,timestamp_s=r['timestamp_ns']/1e9,
                           box_xyxy=r['box'],box_confidence=r['score'],clip=r['clip'],
                           xyz_camera_m=full['base'][i].cpu().tolist(),available_3d=[True]*20,
                           confirmed_3d=[False]*20,predicted_right=int(obs['right'][i])))
    inputpath=out/'runtime_smoke_input.json';outputpath=out/'runtime_smoke_output.json'
    save(inputpath,dict(image_size=[1408,1408],tracks=[dict(id='real_rgb_smoke',frames=frames)]))
    command=[sys.executable,str(Path(__file__).parent/'complete_hand_tracks_stable_v42.py'),
             '--input',str(inputpath),'--output',str(outputpath),'--device',device,'--steps','350']
    # Reconstruct WiLoR from real RGB/boxes rather than passing cached XYZ.
    subprocess.run(command,check=True)
    runtime=json.loads(outputpath.read_text());assert runtime['constraint_checks_passed'] and runtime['rgb_conditioned']
    assert not runtime['tracks'][0]['manual_anchor_blocked'] and runtime['raw_xyz_fallback_frames']==0
    assert len(runtime['tracks'][0]['frames'])==16
    result=dict(passed=True,so3_edge_cases=True,single_frame=True,short_gap_densified=True,long_gap_split=True,
        duplicate_timestamp_rejected=True,gt_poison_exact=True,saved_parameter_xyz_parity=True,chirality_parameter_interpolation=True,
        manual_anchor_conflict_detected=True,real_rgb_raw_boxes_wilor_parameter_head_solver_runtime=True,
        runtime_frames=16,runtime_constraints=runtime['tracks'][0]['constraints'],
        scope='Runtime smoke on existing development images; not another independent accuracy evaluation')
    save(out/'verification.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
