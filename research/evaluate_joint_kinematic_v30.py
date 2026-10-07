"""Frozen proposal/geometry/time study; choose weights on dev_select only."""
import argparse,hashlib,json,time,collections
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,metrics,score,save
from calibrate_hand3d_v8 import paired_ci
from joint_mano_model_v29 import fuse_overlapping_trajectories,EDGES
from joint_trajectory_solver_v30 import fit_trajectory
from kinematic_hand_model_v30 import KinematicHandDecoder

OUT=V7.parent/'joint_kinematic_v30';SOURCE=V7.parent/'joint_mano_v28'
LIMITS=dict(root_speed=3.,pose_speed=3.,root_acc=100.,pose_acc=200.,max_bend_degrees=150.)

def subset(values,ids,n,device):
    return {k:v[ids].to(device) if torch.is_tensor(v) and len(v)==n else v for k,v in values.items()}

def identities(rows,labels):
    """Evaluation metadata only; never passed to fit_trajectory."""
    files={};out=[]
    for i,r in enumerate(rows):
        side=None
        if labels['valid'][i].all():
            sp,seq,clip=Path(r['image']).relative_to('/mnt/why/HOT3D/export/images').parts[:3]
            path=V7.parent.parent/'export/annotations'/sp/seq/(clip+'.jsonl')
            if path not in files:files[path]=[json.loads(x) for x in path.read_text().splitlines()]
            for hand in files[path][r['frame']]['hands']:
                xyz=torch.tensor(hand['xyz_camera_m'])
                if torch.isfinite(xyz).all() and (xyz-labels['gt'][i]).abs().max()<2e-6:
                    assert side is None;side=hand['side']
        out.append(side)
    return out

def coherence(pred,cache,gt,valid,rows,sides):
    groups=collections.defaultdict(list)
    for n,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(n)
    pairs=[];switches=0
    for ns in groups.values():
        ns.sort(key=lambda n:rows[n]['timestamp_ns'])
        for a,b in zip(ns,ns[1:]):
            if rows[b]['frame']-rows[a]['frame']!=1 or sides[a] is None or sides[b] is None:continue
            if sides[a]!=sides[b]:switches+=1;continue
            pairs.append((a,b))
    a,b=map(torch.tensor,zip(*pairs));a=a.to(pred.device);b=b.to(pred.device)
    R,T=cache['rotation'],cache['translation'];world=lambda x:torch.einsum('njc,nkc->njk',x,R)+T[:,None]
    x=world(pred);g=world(gt);xp=x-x[:,5:6];gp=g-g[:,5:6]
    move=(xp[b]-xp[a]).norm(dim=-1)*1000;gm=(gp[b]-gp[a]).norm(dim=-1)*1000
    mask=valid[a]&valid[b];mask[:,5]=False;jump=mask&(gm<10)&(move>30)
    i,j=zip(*EDGES);length=(pred[:,i]-pred[:,j]).norm(dim=-1)*1000;gl=(gt[:,i]-gt[:,j]).norm(dim=-1)*1000;ev=valid[:,i]&valid[:,j];ratio=length/gl.clamp_min(1e-6)
    extreme=((ratio<.5)|(ratio>2))&ev;change=(length[b]-length[a]).abs();gchange=(gl[b]-gl[a]).abs();vm=ev[a]&ev[b]
    return dict(adjacent_same_hand_pairs=len(pairs),identity_switches=switches,spurious_jump_pairs=int(jump.any(-1).sum()),spurious_jump_points=int(jump.sum()),
        bone_extreme_frames=int(extreme.any(-1).sum()),bone_extreme_edges=int(extreme.sum()),bone_length_change_mean_mm=float(change[vm].mean()),
        bone_flicker_gt_under1_pred_over10=int(((change>10)&(gchange<1)&vm).sum()),bone_length_error_mean_mm=float((length-gl).abs()[ev].mean()))

def report(pred,cache,labels,rows,sides,original,accepted=None):
    device=pred.device;gt=labels['gt'].to(device);valid=labels['valid'].to(device)
    centers=torch.tensor([r['window_index'] is not None for r in rows],device=device)
    m=metrics(pred[centers],original[centers],gt[centers],valid[centers]);mask=valid[centers].clone();mask[:,5]=False
    base=original[centers];g=gt[centers]
    error=(((base-base[:,5:6])-(g-g[:,5:6])).norm(dim=-1)*1000*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    hard=error>40;co=coherence(pred,cache,gt,valid,rows,sides)
    result=dict(metrics=m,hard=metrics(pred[centers][hard],base[hard],g[hard],valid[centers][hard]),hard_windows=int(hard.sum()),coherence=co,
        paired_vs_original=paired_ci(pred[centers],base,g,valid[centers],[r for r in rows if r['window_index'] is not None]))
    if accepted is not None:result.update(accepted_frames=int(accepted.sum()),frames=len(rows),accepted_valid_frames=int((accepted&valid.all(-1)).sum()))
    return result

def eligible(candidate,baseline):
    m,b=candidate['metrics'],baseline['metrics'];c,bc=candidate['coherence'],baseline['coherence'];_,ok=score(m)
    conditions=dict(locked_recovery_protection=ok,relative_paired_ci=candidate['paired_vs_original']['relative']['ci95_delta_mm'][1]<0,
        camera_noninferior=m['camera_mm']<=b['camera_mm']+.1,relative_noninferior=m['relative_mm']<=b['relative_mm']+.1,
        hard_recovery=candidate['hard']['relative_bad_recovered20']>=baseline['hard']['relative_bad_recovered20'],
        hard_mean=candidate['hard']['relative_mm']<=baseline['hard']['relative_mm']+.1,
        jump_reduction=c['spurious_jump_pairs']<=bc['spurious_jump_pairs']*.7,
        bone_flicker_reduction=c['bone_flicker_gt_under1_pred_over10']<=bc['bone_flicker_gt_under1_pred_over10']*.7,
        bone_extreme_no_increase=c['bone_extreme_frames']<=bc['bone_extreme_frames'])
    return all(conditions.values()),conditions

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['select','calibrate']);ap.add_argument('--device',default='cuda:0');ap.add_argument('--steps',type=int,default=300);a=ap.parse_args();torch.set_num_threads(4)
    rows=json.loads((SOURCE/'rows.json').read_text());n=len(rows);cache=torch.load(SOURCE/'candidates.pt',weights_only=False,mmap=True);obs=torch.load(SOURCE/'mano_observations.pt',weights_only=False,mmap=True)
    inp=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True)
    # Pool inputs before any label file is loaded.
    cache['fusion_world']=fuse_overlapping_trajectories(cache,inp,rows)
    role='dev_select' if a.stage=='select' else 'dev_calibrate';ids=torch.tensor([i for i,r in enumerate(rows) if r['role']==role]);rr=[rows[i] for i in ids]
    c=subset(cache,ids,n,a.device);o=subset(obs,ids,n,a.device)
    labels_all=torch.load(SOURCE/'evaluation_labels.pt',weights_only=False,mmap=True);labels={k:v[ids] for k,v in labels_all.items()};sides=identities(rr,labels)
    old=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    original=c['base'].clone()
    for i,r in enumerate(rr):
        if r['window_index'] is not None:original[i]=old['original_base_for_evaluation'][r['window_index']].to(a.device)
    baseline=report(c['baseline'],c,labels,rr,sides,original)
    if a.stage=='calibrate':
        selection=json.loads((OUT/'selection.json').read_text());assert selection['approved_on_dev_select'],'Do not open calibration when selection gate failed'
        configs=[selection['selected']['config']]
    else:
        assert not (OUT/'selection.json').exists()
        configs=[]
        for mode,tw,ow,enforce,mix in [('frame',0.,0.,False,0.),('weak',.005,.1,True,0.),('joint',.02,.1,True,0.),('strong',.08,.1,True,0.),('raw',.02,.2,True,.25),('rawhalf',.02,.2,True,.5)]:
            configs.append(dict(name=mode,temporal_weight=tw,overlap_weight=ow,enforce_motion=enforce,rgb_weight=.05,raw_mix=mix,hard_motion_loss=False))
        configs=[dict(**x,steps=a.steps,limits=LIMITS) for x in configs]
    decoder=KinematicHandDecoder(a.device);check=decoder.reconstruction_check(o,c,rr);assert check['passed'],check;save(OUT/(a.stage+'_decoder_check.json'),check)
    results=[];start=time.time()
    print(json.dumps(dict(stage=a.stage,rows=len(rr),baseline=baseline)),flush=True)
    for config in configs:
        path=OUT/(a.stage+'_'+config['name']+'.pt')
        if path.exists():result=torch.load(path,weights_only=False,map_location=a.device)
        else:
            result=fit_trajectory(decoder,c,o,rr,config,progress=lambda x:print(json.dumps(dict(config=config['name'],**x,seconds=time.time()-start)),flush=True))
            torch.save({k:v.cpu() if torch.is_tensor(v) else ({kk:vv.cpu() for kk,vv in v.items()} if k=='parameters' else v) for k,v in result.items()},path)
        m=report(result['prediction'],c,labels,rr,sides,original,result['frame_accepted']);raw=report(result['hypothesis'],c,labels,rr,sides,original)
        ok,conditions=eligible(m,baseline);paired=paired_ci(result['prediction'][torch.tensor([r['window_index'] is not None for r in rr],device=a.device)],c['baseline'][torch.tensor([r['window_index'] is not None for r in rr],device=a.device)],labels['gt'][torch.tensor([r['window_index'] is not None for r in rr])].to(a.device),labels['valid'][torch.tensor([r['window_index'] is not None for r in rr])].to(a.device),[r for r in rr if r['window_index'] is not None])
        entry=dict(config=config,final=m,hypothesis_review_only=raw,eligible=ok,conditions=conditions,paired_vs_v16=paired,accepted_tracks=int(result['accepted'].sum()),tracks=len(result['accepted']))
        results.append(entry);save(OUT/(a.stage+'_partial_results.json'),dict(baseline=baseline,results=results));print(json.dumps(entry),flush=True)
    passing=[x for x in results if x['eligible'] and x['config']['enforce_motion']]
    selected=min(passing,key=lambda x:x['final']['metrics']['camera_mm']+.75*x['final']['metrics']['relative_mm']) if passing else None
    summary=dict(complete=True,stage=a.stage,baseline=baseline,results=results,approved_on_dev_select=selected is not None,selected=selected,default_changed=False,seconds=time.time()-start,
        scope='Existingdevelopmentonly. Selectionweights ondev_select only. Motion/geometry/protection acceptance is wholetrack; fallback included in allmetrics, not hidden. Candidate hypotheses not approved.')
    save(OUT/('selection.json' if a.stage=='select' else 'calibration_results.json'),summary);print(json.dumps(dict(complete=True,approved=selected is not None,seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
