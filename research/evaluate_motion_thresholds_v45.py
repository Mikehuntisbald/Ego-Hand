"""Fixed v44 DiT path: diagnose smoothing and motion bounds separately."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from hand3d_v8_common import V7,save
from parameter_candidates_v43 import load_observations
from parameter_codec_v31 import OUT
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from stability_trajectory_v43 import DEFAULT,saved_check
from motion_threshold_solver_v45 import fit_stable
from evaluate_matched_stability_v43 import compare
from evaluate_joint_kinematic_v30 import identities
from calibrate_hand3d_v8 import paired_ci

KEYS=['root_speed','pose_speed','wrist_speed','joint_speed','root_acc','pose_acc','wrist_acc','joint_acc']


def configs():
    baseline=dict(DEFAULT)
    motion=lambda speed,acc:{k:DEFAULT[k]*(speed if k.endswith('speed') else acc) for k in KEYS}
    return dict(strict=baseline,
        hard_only_x2=dict(baseline,**motion(2,2),optimization_motion_limits={k:DEFAULT[k] for k in KEYS}),
        speed_x2=dict(baseline,**motion(2,1)),acc_x2=dict(baseline,**motion(1,2)),
        motion_x2=dict(baseline,**motion(2,2)),motion_x3=dict(baseline,**motion(3,3)),
        soft_acc_x2=dict(baseline,optimization_motion_limits=motion(1,2)),
        soft_temporal_acc_x2=dict(baseline,temporal_motion_limits=motion(1,2)),
        smooth_45ms=dict(baseline,smoothing_s=.045),
        motion_x2_smooth45=dict(baseline,**motion(2,2),smoothing_s=.045),
        motion_x2_less_soft=dict(baseline,**motion(2,2),smoothing_s=.045,temporal_weight=.02))


def serialize(result):
    return {k:({kk:vv.cpu() for kk,vv in v.items()} if k in ['parameters','stages'] else v.cpu() if torch.is_tensor(v) else v) for k,v in result.items()}


def gt_motion_audit(rows,cache,labels,limits):
    """GT-only audit, loaded after every solver output has frozen."""
    device=cache['base'].device;gt=labels['gt'].to(device);valid=labels['valid'].to(device)
    sides=identities(rows,labels);groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    pairs=[];triples=[]
    for ix in groups.values():
        ix.sort(key=lambda i:rows[i]['timestamp_ns'])
        good=lambda a,b:rows[b]['frame']==rows[a]['frame']+1 and sides[a] is not None and sides[a]==sides[b]
        pairs += [(a,b) for a,b in zip(ix,ix[1:]) if good(a,b)]
        triples += [(a,b,c) for a,b,c in zip(ix,ix[1:],ix[2:]) if good(a,b) and good(b,c)]
    pair=torch.tensor(pairs,device=device);triple=torch.tensor(triples,device=device)
    times=torch.tensor([r['timestamp_ns'] for r in rows],device=device,dtype=torch.float64)/1e9
    a,b=pair.T;ia,ib,ic=triple.T
    world=torch.einsum('njc,nkc->njk',gt,cache['rotation'])+cache['translation'][:,None]
    pose=world-world[:,5:6]
    dt=(times[b]-times[a]).float();d1=(times[ib]-times[ia]).float();d2=(times[ic]-times[ib]).float();h=(d1+d2)/2
    pv=(pose[b]-pose[a])/dt[:,None,None]
    pa=((pose[ic]-pose[ib])/d2[:,None,None]-(pose[ib]-pose[ia])/d1[:,None,None])/h[:,None,None]
    rv=(world[b,5]-world[a,5])/dt[:,None]
    ra=((world[ic,5]-world[ib,5])/d2[:,None]-(world[ib,5]-world[ia,5])/d1[:,None])/h[:,None]
    pm=valid[a]&valid[b]&valid[a,5,None]&valid[b,5,None];pm[:,5]=False
    tm=valid[ia]&valid[ib]&valid[ic]&valid[ia,5,None]&valid[ib,5,None]&valid[ic,5,None];tm[:,5]=False
    files={};angles=torch.zeros(len(rows),20,device=device);rotations=torch.eye(3,device=device)[None].repeat(len(rows),1,1);pose_valid=torch.zeros(len(rows),device=device,dtype=torch.bool)
    from scipy.spatial.transform import Rotation
    for i,r in enumerate(rows):
        if sides[i] is None:continue
        split,seq,clip=Path(r['image']).relative_to('/mnt/why/HOT3D/export/images').parts[:3]
        path=V7.parent.parent/'export/annotations'/split/seq/(clip+'.jsonl')
        if path not in files:files[path]=[json.loads(s) for s in path.read_text().splitlines()]
        hands=[v for v in files[path][r['frame']]['hands'] if v['side']==sides[i]]
        if len(hands)!=1 or 'umetrack_pose' not in hands[0]:continue
        p=hands[0]['umetrack_pose'];angle=torch.tensor(p['joint_angles'][:20],device=device)
        q=p['T_world_from_wrist']['quaternion_wxyz'];R=Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix()
        if not torch.isfinite(angle).all():continue
        angles[i]=angle;rotations[i]=torch.tensor(R,device=device,dtype=torch.float32);pose_valid[i]=True
    jv=(angles[b]-angles[a])/dt[:,None]
    ja=((angles[ic]-angles[ib])/d2[:,None]-(angles[ib]-angles[ia])/d1[:,None])/h[:,None]
    from stability_trajectory_v43 import trajectory_motion
    motion=trajectory_motion(world,rotations,angles,(a,b,dt),(ia,ib,ic,d1,d2))
    pvm=pose_valid[a]&pose_valid[b];tvm=pose_valid[ia]&pose_valid[ib]&pose_valid[ic]
    values=dict(root_speed=rv.norm(dim=-1)[valid[a,5]&valid[b,5]],root_acc=ra.norm(dim=-1)[valid[ia,5]&valid[ib,5]&valid[ic,5]],
        pose_speed=pv.norm(dim=-1)[pm],pose_acc=pa.norm(dim=-1)[tm],
        joint_speed=jv.abs()[pvm].flatten(),joint_acc=ja.abs()[tvm].flatten(),
        wrist_speed=motion['wrist_speed'].norm(dim=-1)[pvm],wrist_acc=motion['wrist_acc'].norm(dim=-1)[tvm])
    result={}
    for name,x in values.items():
        result[name]=dict(limit=limits[name],count=len(x),median=float(x.median()),p90=float(x.quantile(.9)),p95=float(x.quantile(.95)),p99=float(x.quantile(.99)),
            above_limit=int((x>limits[name]).sum()),above_limit_fraction=float((x>limits[name]).float().mean()))
    step=(pose[b]-pose[a]).norm(dim=-1)*1000;fast=pm&(step>10);speed=pv.norm(dim=-1)
    result['fast_pose_speed']=dict(count=int(fast.sum()),median=float(speed[fast].median()),above_pose_speed_limit=int((fast&(speed>limits['pose_speed'])).sum()),
        note='Unfiltered mocap finite differences include annotation jitter; exceeding a cap is diagnostic, not an inference instruction.')
    result['valid_pairs']=len(pairs);result['valid_triples']=len(triples)
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--variants',nargs='+',default=list(configs()))
    ap.add_argument('--evaluate-only',action='store_true');a=ap.parse_args();torch.set_num_threads(4)
    root=V7.parent;out=root/'motion_threshold_v45';out.mkdir(exist_ok=True);start=time.time()
    rows,ids,_,cache=load_observations(a.device)
    samples=torch.load(root/'short_context_v44/dit/candidates.pt',weights_only=False)
    path=json.loads((root/'short_context_v44/dit/sequence_selection.json').read_text())['selected_index']
    ii=torch.arange(len(rows));kk=torch.tensor(path)
    observation=dict(xyz=samples['xyz'][ii,kk],parameter_state=samples['states'][ii,kk],right=samples['right'])
    side=torch.load(OUT/'predicted_right_bank.pt',weights_only=False)
    observation['parameter_side_outlier']=observation['right']!=side[torch.tensor([r['center_fid'] for r in rows])]
    obs={k:v.to(a.device) for k,v in observation.items()};cache=dict(cache,risk=samples['risk'].to(a.device))
    decoder=ParameterTrajectoryDecoder(a.device)
    save(out/'protocol.json',dict(experiment='Fixed DiT candidate/path motion-bound and smoothing diagnosis',configs=configs(),
        source_candidates_sha256=hashlib.sha256((root/'short_context_v44/dit/candidates.pt').read_bytes()).hexdigest(),
        selected_path_sha256=hashlib.sha256((root/'short_context_v44/dit/sequence_selection.json').read_bytes()).hexdigest(),
        same_dit_checkpoint='matched_parameter_v43/dit selectedstep3000',same_context_s=1.6,
        same_hand_decoder=True,same_shape_and_chirality=True,no_gt_solver_or_path_selection=True,
        default_changed=False,scope='Previously reused development; no new-subject evidence. Selector remains fixed to isolate solver constraints.'))
    if not a.evaluate_only:
        for name in a.variants:
            assert name in configs();resultpath=out/(name+'.pt')
            if resultpath.exists():continue
            result=fit_stable(decoder,cache,obs,rows,configs()[name],progress=lambda x:print(json.dumps(dict(variant=name,**x,seconds=time.time()-start)),flush=True))
            torch.save(serialize(result),resultpath)
            saved=torch.load(resultpath,weights_only=False);pp={k:v.to(a.device) for k,v in saved['parameters'].items()}
            world,check=saved_check(decoder,pp,saved['layout'],configs()[name]);assert check['passed']
            assert torch.allclose(world.cpu(),saved['world'],atol=1e-7,rtol=0)
            source=torch.tensor(saved['layout']['source_map'],device=a.device)
            camera=torch.einsum('njc,nck->njk',world[source]-cache['translation'][:,None],cache['rotation'])
            assert torch.allclose(camera.cpu(),saved['prediction'],atol=1e-7,rtol=0)
            if name=='strict':
                old=torch.load(root/'short_context_v44/dit/sequence.pt',weights_only=False)['prediction']
                parity=float((old-saved['prediction']).abs().max());assert parity==0.,parity
            else:parity=None
            save(out/(name+'_check.json'),dict(passed=True,saved_parameter_redecode=check,strict_parity_max_m=parity))
            print(json.dumps(dict(variant=name,done=True,seconds=time.time()-start)),flush=True)
    if not all((out/(name+'.pt')).exists() for name in configs()):return
    save(out/'freeze.json',dict(all_outputs_frozen_before_evaluation=True,sha256={n:hashlib.sha256((out/(n+'.pt')).read_bytes()).hexdigest() for n in configs()}))
    strict=torch.load(out/'strict.pt',weights_only=False)
    predictions=dict(v42=strict['prediction'].to(a.device),raw_selected=obs['xyz'])
    predictions.update({k:v.to(a.device) for k,v in strict['stages'].items()})
    coverage={}
    for name in configs():
        saved=torch.load(out/(name+'.pt'),weights_only=False);predictions[name]=saved['prediction'].to(a.device)
        restoration=saved['restoration'];rr=np.asarray([v['pose_scale'] for v in restoration]);rc=np.asarray([v['root_scale'] for v in restoration])
        coverage[name]=dict(segments=len(restoration),pose_contracted_segments=int((rr<1).sum()),pose_scale_median=float(np.median(rr)),
            pose_scale_min=float(rr.min()),pose_scale_under_half_segments=int((rr<=.5).sum()),constant_pose_segments=int((rr==0).sum()),
            root_contracted_segments=int((rc<1).sum()),root_scale_min=float(rc.min()))
    reports,diagnostics,tensors=compare(predictions,cache,rows)
    labels_all=torch.load(root/'joint_mano_v28/evaluation_labels.pt',weights_only=False,mmap=True);labels={k:v[ids] for k,v in labels_all.items()}
    gt,valid=labels['gt'].to(a.device),labels['valid'].to(a.device);centers=torch.tensor([r['window_index'] is not None for r in rows],device=a.device);cr=[r for r in rows if r['window_index'] is not None]
    base=predictions['strict'];error=lambda p:((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
    be=error(base);mask=valid.clone();mask[:,5]=False;good=mask&(be<=10);bad=mask&(be>20)
    protection={}
    for name,p in predictions.items():
        e=error(p);protection[name]=dict(strict_good_points=int(good.sum()),strict_good_to_bad_points=int((good&(e>20)).sum()),
            strict_bad_points=int(bad.sum()),strict_bad_to_good_points=int((bad&(e<=20)).sum()),
            paired_vs_strict=paired_ci(p[centers],base[centers],gt[centers],valid[centers],cr))
    audit=gt_motion_audit(rows,cache,labels,DEFAULT)
    summary=dict(complete=True,reports=reports,diagnostics=diagnostics,coverage=coverage,protection=protection,gt_motion_audit=audit,
        configs=configs(),strict_parity_max_m=0.,default_changed=False,seconds=time.time()-start,
        scope='2673 reused observations,352 development centers,4P0003sequences; fixed DiT path. GT motion statistics are diagnosis only; not used by the solver.')
    save(out/'summary.json',summary);torch.save(tensors,out/'audit_tensors.pt');save(out/'rows.json',rows)
    print(json.dumps(dict(complete=True,table={k:dict(relative=v['metrics']['relative_mm'],camera=v['metrics']['camera_mm'],
        fast=diagnostics[k]['fast_amplitude_ratio']['median'],direction=diagnostics[k]['fast_directional_retention']['median'],
        jumps=v['coherence']['spurious_jump_pairs'],harm=protection[k]['strict_good_to_bad_points']) for k,v in reports.items()},gt_audit=audit)),flush=True)


if __name__=='__main__':main()
