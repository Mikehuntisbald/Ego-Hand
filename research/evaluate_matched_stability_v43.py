"""Frozen matched-generator trials through exactly the released v42 solver."""
import argparse,collections,hashlib,json,time
import torch
from hand3d_v8_common import V7,save
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from parameter_candidates_v43 import generate,load_observations,select_sequence
from stability_trajectory_v42 import DEFAULT,fit_stable,saved_check
from evaluate_stability_v42 import quantiles
from evaluate_joint_kinematic_v30 import report,identities
from calibrate_hand3d_v8 import paired_ci


def compare(predictions,cache,rows):
    source=V7.parent/'joint_mano_v28';allrows=json.loads((source/'rows.json').read_text())
    ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select']);device=cache['base'].device
    la=torch.load(source/'evaluation_labels.pt',weights_only=False,mmap=True);labels={k:v[ids] for k,v in la.items()}
    gt,valid=labels['gt'].to(device),labels['valid'].to(device);sides=identities(rows,labels)
    dense=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    original=cache['base'].clone()
    for i,r in enumerate(rows):
        if r['window_index'] is not None:original[i]=dense['original_base_for_evaluation'][r['window_index']].to(device)
    reports={name:report(p,cache,labels,rows,sides,original) for name,p in predictions.items()}
    groups=collections.defaultdict(list);pairs=[]
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    for ix in groups.values():
        ix.sort(key=lambda i:rows[i]['timestamp_ns'])
        for a,b in zip(ix,ix[1:]):
            if rows[b]['frame']-rows[a]['frame']==1 and sides[a] is not None and sides[a]==sides[b]:pairs.append((a,b))
    a,b=map(lambda x:torch.tensor(x,device=device),zip(*pairs));mask=valid.clone();mask[:,5]=False
    pm=valid[a]&valid[b];pm[:,5]=False
    world=lambda p:torch.einsum('njc,nkc->njk',p,cache['rotation'])+cache['translation'][:,None]
    gw=world(gt);gp=gw-gw[:,5:6];gd=gp[b]-gp[a];gm=gd.norm(dim=-1)*1000;fast=pm&(gm>10)
    centers=torch.tensor([r['window_index'] is not None for r in rows],device=device)
    center_rows=[r for r in rows if r['window_index'] is not None]
    old=predictions['v42'];old_error=((old-old[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
    old_good=mask&(old_error<=10);old_bad=mask&(old_error>20)
    extras={};tensors={}
    for name,p in predictions.items():
        pw=world(p);pose=pw-pw[:,5:6];delta=pose[b]-pose[a];move=delta.norm(dim=-1)*1000
        error=((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
        retention=move/gm.clamp_min(1e-6);direction=(delta*gd).sum(-1)/gd.square().sum(-1).clamp_min(1e-10)
        extras[name]=dict(all_valid_relative_mm=quantiles(error[mask]),relative_step_mm=quantiles(move[pm]),
            fast_point_pairs=int(fast.sum()),fast_amplitude_ratio=quantiles(retention[fast]),
            fast_directional_retention=quantiles(direction[fast]),fast_under_half_points=int((fast&(retention<.5)).sum()),
            v42_good_points=int(old_good.sum()),v42_good_to_bad_points=int((old_good&(error>20)).sum()),
            v42_bad_points=int(old_bad.sum()),v42_bad_to_good_points=int((old_bad&(error<=20)).sum()),
            paired_center_vs_v42=paired_ci(p[centers],old[centers],gt[centers],valid[centers],center_rows))
        tensors[name]=dict(relative_error=error.cpu(),world=pw.cpu(),move=move.cpu())
    return reports,extras,tensors


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['regression','dit','dit_denoise'],required=True)
    ap.add_argument('--device',default='cuda:3');ap.add_argument('--consistent-initial-side',action='store_true')
    a=ap.parse_args();torch.set_num_threads(4)
    root=V7.parent;out=root/'matched_stability_v43'/(a.kind+'_side_consistent' if a.consistent_initial_side else a.kind);out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('Do not replace a completed frozen comparison')
    start=time.time();rows,_,_,cache=load_observations(a.device)
    samplepath=root/'matched_stability_v43'/a.kind/'candidates.pt'
    samplepath.parent.mkdir(parents=True,exist_ok=True)
    if samplepath.exists():samples=torch.load(samplepath,weights_only=False)
    else:
        samples=generate('matched_parameter_v43/'+a.kind,a.device,progress=lambda x:print(json.dumps(dict(stage='sample',kind=a.kind,**x)),flush=True))
        torch.save(samples,samplepath)
    observations={'mean':samples['mean_observation']};selection=None
    if a.kind.startswith('dit'):
        obs,selection=select_sequence(samples,cache,rows);observations['sequence']=obs;save(out/'sequence_selection.json',selection)
        # GT-poison invariance also covers discrete sequence selection.
        poisoned=dict(cache,gt=torch.full_like(cache['base'],float('nan')),valid=torch.zeros(len(rows),20,device=a.device,dtype=torch.bool))
        other,check=select_sequence(samples,poisoned,rows)
        assert check['selected_index']==selection['selected_index'] and torch.equal(other['parameter_state'],obs['parameter_state'])
    if a.consistent_initial_side:
        from stability_trajectory_v43 import fit_stable as solve
        from parameter_codec_v31 import OUT
        sidebank=torch.load(OUT/'predicted_right_bank.pt',weights_only=False)
        original_side=sidebank[torch.tensor([r['center_fid'] for r in rows])]
        for observation in observations.values():observation['parameter_side_outlier']=observation['right']!=original_side
    else:solve=fit_stable
    save(out/'protocol.json',dict(kind=a.kind,generator=samples['model_run'],selected_step=samples['step'],
        model_kind_only_comparison='Matched regression andDiT use same architecture, initialweights, data order, budget, seed and checkpoint rule',
        common_solver='v42 objective, decoder, limits, schedule, restoration; common seed/head-side prefilter' if a.consistent_initial_side else 'Released stability_trajectory_v42 with unchanged DEFAULT; shared frozen WiLoR/RGB candidate evidence',
        initial_side_consistency=a.consistent_initial_side,
        config=DEFAULT,samples=samples['draws'],ddim_steps=samples['ddim_steps'],candidate_selection='Meanparameter decoding and label-free second-order whole-segment path' if a.kind.startswith('dit') else 'Deterministic regression',
        target_gt_inference=False,calibration_performance_evaluated=False,default_changed=False,
        scope='Previously used 2673dev_select detections/352centers, 4sequencesP0003; no new subject validation'))
    decoder=ParameterTrajectoryDecoder(a.device);frozen={};coverage={}
    for name,observation in observations.items():
        obs={k:v.to(a.device) for k,v in observation.items()}
        parity=decoder.reconstruction_check(obs,cache,rows);assert parity['passed'];save(out/f'{name}_decoder_check.json',parity)
        result=solve(decoder,cache,obs,rows,DEFAULT,progress=lambda x:print(json.dumps(dict(stage='solver',kind=a.kind,variant=name,**x)),flush=True))
        saveable={k:({kk:vv.cpu() for kk,vv in v.items()} if k=='parameters' else v.cpu() if torch.is_tensor(v) else v) for k,v in result.items()}
        torch.save(saveable,out/(name+'.pt'))
        saved=torch.load(out/(name+'.pt'),weights_only=False);p={k:v.to(a.device) for k,v in saved['parameters'].items()}
        xyz,check=saved_check(decoder,p,saved['layout'],saved['config']);assert check['passed']
        assert torch.allclose(xyz.cpu(),saved['world'],atol=1e-7,rtol=0)
        check['saved_parameter_redecode']=True;save(out/(name+'_constraint_check.json'),check)
        restoration=result['restoration'];coverage[name]=dict(check=check,pose_scale=quantiles([x['pose_scale'] for x in restoration]),
            root_scale=quantiles([x['root_scale'] for x in restoration]),constant_pose_segments=sum(x['constant_pose_fallback'] for x in restoration),
            pose_scale_under_half_segments=sum(x['pose_scale']<=.5 for x in restoration),side_outlier_frames=result['side_parameter_outlier_frames'])
        frozen[name]=result['prediction']
    save(out/'freeze.json',dict(all_outputs_frozen=True,labels_read_after_freeze=True,
         results_sha256={name:hashlib.sha256((out/(name+'.pt')).read_bytes()).hexdigest() for name in frozen}))
    reference=torch.load(root/'stability_first_v42_sidefix/results.pt',weights_only=False)['prediction'].to(a.device)
    if a.consistent_initial_side:
        ro=torch.load(root/'fitted_parameter_trajectory_v39/observations.pt',weights_only=False)
        ro['parameter_side_outlier']=ro['right']!=original_side
        rr=solve(decoder,cache,{k:v.to(a.device) for k,v in ro.items()},rows,DEFAULT)
        torch.save({k:({kk:vv.cpu() for kk,vv in v.items()} if k=='parameters' else v.cpu() if torch.is_tensor(v) else v) for k,v in rr.items()},out/'reference_v42.pt')
        save(out/'reference_v42_parity.json',dict(mean_mm=float((rr['prediction']-reference).norm(dim=-1).mean()*1000),max_mm=float((rr['prediction']-reference).norm(dim=-1).max()*1000),same_constraints=True))
        reference=rr['prediction']
    variants=dict(v42=reference,**{a.kind+'_'+k:v for k,v in frozen.items()})
    reports,extra,tensors=compare(variants,cache,rows)
    summary=dict(complete=True,kind=a.kind,step=samples['step'],inference_seconds=samples['seconds'],seconds=time.time()-start,
        center_reports=reports,diagnostics=extra,coverage=coverage,
        gt_poison_generator_exact=samples['gt_poison_exact'],gt_poison_selector_exact=selection is not None,
        default_changed=False,scope='Matched single-seed development trial; all point/motion safeguards same as v42; no fresh generalization claim')
    save(out/'summary.json',summary);torch.save(tensors,out/'audit_tensors.pt');save(out/'rows.json',rows)
    print(json.dumps(dict(complete=True,kind=a.kind,step=samples['step'],seconds=time.time()-start,
        summary={k:dict(camera=v['metrics']['camera_mm'],relative=v['metrics']['relative_mm'],coherence=v['coherence']) for k,v in reports.items()})),flush=True)


if __name__=='__main__':main()
