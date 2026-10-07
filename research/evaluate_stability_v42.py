"""Freeze all stability-first outputs before posthoc development evaluation."""
import argparse,collections,hashlib,json,time
import numpy as np
import torch
from hand3d_v8_common import V7,save
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from evaluate_joint_kinematic_v30 import subset,identities,report
from joint_mano_model_v29 import EDGES
from stability_trajectory_v42 import fit_stable,DEFAULT,saved_check


def quantiles(x):
    x=torch.as_tensor(x).float().flatten();x=x[torch.isfinite(x)]
    return dict(n=len(x),mean=float(x.mean()),p05=float(x.quantile(.05)),median=float(x.median()),p95=float(x.quantile(.95)),max=float(x.max())) if len(x) else dict(n=0)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--steps',type=int,default=350)
    ap.add_argument('--name',default='stability_first_v42');a=ap.parse_args();torch.set_num_threads(4)
    root=V7.parent;source=root/'joint_mano_v28';out=root/a.name;out.mkdir(exist_ok=True)
    if (out/'results.pt').exists():raise RuntimeError('Choose a fresh run name; do not overwrite frozen results')
    start=time.time();allrows=json.loads((source/'rows.json').read_text());ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select']);rows=[allrows[i] for i in ids]
    cache=subset(torch.load(source/'candidates.pt',weights_only=False,mmap=True),ids,len(allrows),a.device)
    obs={k:v.to(a.device) for k,v in torch.load(root/'fitted_parameter_trajectory_v39/observations.pt',weights_only=False).items()}
    config=dict(DEFAULT,steps=a.steps)
    save(out/'protocol.json',dict(objective='Plausible bounded hand articulation and stable offline motion take priority over exact keypoint accuracy',
        generator='Frozen fitted_parameter_v39/regression, selected1750; full spatial RGB and bidirectional temporal observations',
        data='Previously used dev_select only, 4sequences one subjectP0003; no new accuracy generalization evidence',
        inference_labels=False,artificial_occlusion=False,raw_xyz_fallback=False,
        static_shape='Train-only PCA, median per predicted track, coefficient cap2',
        motion_prior='Observed parameter prior plus velocity/acceleration regularization; HMP learned prior NOT integrated',
        hard_point_preservation_caps=False,reason='User explicitly changed objective to stable plausible estimates',
        config=config,default_mode='Separate selectable stability mode; retain v16 accuracy mode'))
    decoder=ParameterTrajectoryDecoder(a.device)
    result=fit_stable(decoder,cache,obs,rows,config,progress=lambda x:print(json.dumps(dict(stage='fit',**x)),flush=True))
    frozen={k:({kk:vv.cpu() for kk,vv in v.items()} if k=='parameters' else v.cpu() if torch.is_tensor(v) else v) for k,v in result.items()}
    torch.save(frozen,out/'results.pt');save(out/'constraint_check.json',result['check']);save(out/'restoration.json',result['restoration'])
    # Reload saved parameters: a check of optimizer tensors alone is insufficient.
    reload=torch.load(out/'results.pt',weights_only=False);parameters={k:v.to(a.device) for k,v in reload['parameters'].items()}
    world,check=saved_check(decoder,parameters,reload['layout'],reload['config'])
    assert check['passed'];assert torch.allclose(world.cpu(),reload['world'],atol=1e-7,rtol=0)
    check['saved_parameter_redecode_exact']=True;save(out/'constraint_check.json',check)
    save(out/'freeze.json',dict(results_sha256=hashlib.sha256((out/'results.pt').read_bytes()).hexdigest(),labels_read_after_this_freeze=True))
    print(json.dumps(dict(stage='frozen',check=check)),flush=True)
    # Only now access annotation labels and GT identities.
    la=torch.load(source/'evaluation_labels.pt',weights_only=False,mmap=True);labels={k:v[ids] for k,v in la.items()}
    side=identities(rows,labels);gt=labels['gt'].to(a.device);valid=labels['valid'].to(a.device)
    data=torch.load(root/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True);original=cache['base'].clone()
    for i,r in enumerate(rows):
        if r['window_index'] is not None:original[i]=data['original_base_for_evaluation'][r['window_index']].to(a.device)
    v39=torch.load(root/'fitted_joint_refinement_v39/results.pt',weights_only=False)['prediction'].to(a.device)
    variants=dict(v16=cache['baseline'],v39=v39,v42=result['prediction'])
    reports={k:report(p,cache,labels,rows,side,original) for k,p in variants.items()}
    tracks=collections.defaultdict(list);pairs=[]
    for i,r in enumerate(rows):tracks[(r['sequence'],r['clip'],r['track_id'])].append(i)
    for ix in tracks.values():
        ix.sort(key=lambda i:rows[i]['timestamp_ns'])
        for l,r in zip(ix,ix[1:]):
            if rows[r]['frame']-rows[l]['frame']==1 and side[l] is not None and side[l]==side[r]:pairs.append((l,r))
    l,r=map(lambda v:torch.tensor(v,device=a.device),zip(*pairs));pm=valid[l]&valid[r];pm[:,5]=False
    R,T=cache['rotation'],cache['translation'];toworld=lambda x:torch.einsum('njc,nkc->njk',x,R)+T[:,None]
    gp=toworld(gt);gd=(gp-gp[:,5:6])[r]-(gp-gp[:,5:6])[l];gm=gd.norm(dim=-1)*1000
    fast=pm&(gm>10);stats={};curves={}
    mask=valid.clone();mask[:,5]=False
    for name,p in variants.items():
        wp=toworld(p);pose=wp-wp[:,5:6];pd=pose[r]-pose[l];move=pd.norm(dim=-1)*1000
        ce=(p-gt).norm(dim=-1)*1000;re=((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
        retention=move/gm.clamp_min(1e-6);projected=(pd*gd).sum(-1)/gd.square().sum(-1).clamp_min(1e-10)
        stats[name]=dict(camera_all_valid_mm=quantiles(ce[mask]),relative_all_valid_mm=quantiles(re[mask]),
            relative_step_mm=quantiles(move[pm]),real_fast_point_pairs=int(fast.sum()),
            fast_motion_amplitude_ratio=quantiles(retention[fast]),fast_motion_directional_retention=quantiles(projected[fast]),
            fast_motion_amplitude_under_half=int((fast&(retention<.5)).sum()),
            camera_root_step_mm=quantiles((wp[r,5]-wp[l,5]).norm(dim=-1)*1000))
        curves[name]=dict(relative_error=re.detach().cpu(),world=wp.detach().cpu(),move=move.detach().cpu())
    old_good=mask&(curves['v16']['relative_error'].to(a.device)<=10)
    new_error=curves['v42']['relative_error'].to(a.device)
    costs=dict(v16_relative_good_points=int(old_good.sum()),v16_good_to_v42_over20_points=int((old_good&(new_error>20)).sum()))
    restoration=result['restoration'];coverage=dict(observations=len(rows),dense_parameter_frames=len(result['world']),
        interpolated_parameter_frames=len(result['world'])-len(rows),tracks=len(tracks),continuous_segments=len(restoration),
        raw_xyz_fallback_frames=0,constant_pose_segments=sum(x['constant_pose_fallback'] for x in restoration),
        pose_scale=quantiles([x['pose_scale'] for x in restoration]),root_scale=quantiles([x['root_scale'] for x in restoration]),
        pose_scale_at_most_half_segments=sum(x['pose_scale']<=.5 for x in restoration),manual_anchor_conflicts=int(result['manual_anchor_conflicts'].sum()))
    coverage['side_parameter_outlier_frames']=result['side_parameter_outlier_frames']
    summary=dict(complete=True,stability_checks_passed=check['passed'],coverage=coverage,config=config,
        center_reports=reports,all_frame_diagnostic=stats,accuracy_tradeoff=costs,seconds=time.time()-start,
        selection_scope='Previously reused development, 4 sequencesP0003, 352 labeled center windows, 2673observations. No new subject test.',
        guarantee_scope=check['guarantee_scope'],default_accuracy_mode_changed=False,stability_mode_available=True,
        limitations='No self-collision guarantee; pose estimates under occlusion are plausible hypotheses, not recovered truth. Gaps >0.55s split, IDs not reassociated. Motion contraction and wrong depth still possible.')
    save(out/'summary.json',summary);torch.save(curves,out/'audit_tensors.pt');save(out/'rows.json',rows)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
