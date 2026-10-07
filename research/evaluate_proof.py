"""Locked P0010/P0015 test, point-level difficulty groups, clustered comparisons."""
import json,time
import numpy as np,torch
from train_coarse_pose import RUN,ROOT
from residual_models import ResidualModel
from temporal_residual import TemporalResidualModel,attach_context
from metrics_3d import EVAL_INDICES,compare

CHAINS=[[6,7,0],[8,9,10,1],[11,12,13,2],[14,15,16,3],[17,18,19,4]]
def ray_masks(gt,valid):
    masks=np.zeros((len(gt),20),bool);counts=np.zeros(len(gt),int)
    for chain in CHAINS:
        a,b=chain[0],chain[-1];axis=gt[:,b]-gt[:,a];ray=(gt[:,b]+gt[:,a])/2
        cos=np.abs((axis*ray).sum(-1))/(np.linalg.norm(axis,axis=-1)*np.linalg.norm(ray,axis=-1)+1e-9)
        aligned=(cos>=np.cos(np.deg2rad(15)))&valid[:,a]&valid[:,b]
        masks[:,chain]|=aligned[:,None];counts+=aligned
    return masks,counts

def errors(pred,gt):
    ray=gt/(np.linalg.norm(gt,axis=-1,keepdims=True)+1e-9)
    difference=pred-gt;relative=(pred-pred[:,5:6])-(gt-gt[:,5:6])
    return dict(camera=np.linalg.norm(difference,axis=-1)*1000,
                relative=np.linalg.norm(relative,axis=-1)*1000,
                axial_relative=np.abs((relative*ray).sum(-1))*1000)

def cluster_ci(delta,mask,clusters):
    unique=np.unique(clusters);valid=[]
    for c in unique:
        m=mask&(clusters==c)[:,None]
        if m.sum():valid.append((delta[m].sum(),int(m.sum())))
    if len(valid)<2:return dict(clusters=len(valid),ci95=None)
    rng=np.random.default_rng(20261003);values=[]
    for _ in range(2000):
        selected=[valid[i] for i in rng.integers(len(valid),size=len(valid))]
        values.append(sum(s for s,n in selected)/sum(n for s,n in selected))
    return dict(clusters=len(valid),ci95=np.quantile(values,[.025,.975]).tolist(),unit='source sequence; only two test subjects')

def group_report(pred,gt,coarse,mask,clusters):
    a=errors(coarse,gt);b=errors(pred,gt);result=dict(joints=int(mask.sum()),hands=int(mask.any(-1).sum()))
    if not mask.sum():return result
    for metric in a:
        before=float(a[metric][mask].mean());after=float(b[metric][mask].mean())
        result[metric]=dict(before_mm=before,after_mm=after,change_mm=after-before,
            improvement_pct=100*(before-after)/max(1e-9,before),**cluster_ci(b[metric]-a[metric],mask,clusters))
    correct=(a['relative']<=10)&mask;harmed=(b['relative']>a['relative']+1)&correct
    result['correct_relative_joints']=int(correct.sum())
    result['correct_relative_joints_harmed_fraction']=float(harmed.sum()/max(1,correct.sum()))
    result['pck_relative10_before']=float((a['relative'][mask]<=10).mean());result['pck_relative10_after']=float((b['relative'][mask]<=10).mean())
    return result

def infer(model,data,ids,device='cuda:0',calibration=None):
    outputs=[];ungated=[];gate_values=[];started=time.time();generator=torch.Generator(device=device).manual_seed(901003)
    with torch.inference_mode():
        for start in range(0,len(ids),256):
            ix=ids[start:start+256];c=data['coarse'][ix].to(device);conf=data['confidence'][ix].to(device);rgb=data['rgb'][ix].to(device).float()
            attach_context(model,data,ix,device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                proposal=model.propose(c,conf,rgb,steps=10,samples=4 if model.kind=='dit' else 1,generator=generator)
                _,gates=model.gates(proposal,c,conf,rgb)
                if calibration is not None:
                    gates=gates*(gates>=calibration['threshold'])*calibration['strength']
                    pred=model.apply_gates(proposal,c,gates)
                else:pred=model.unpack(model.pack(c)+gates[...,None]*proposal)
                raw=model.unpack(model.pack(c)+proposal)
            outputs.append(pred.float().cpu().numpy());ungated.append(raw.float().cpu().numpy());gate_values.append(gates.float().cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(outputs),np.concatenate(ungated),np.concatenate(gate_values),time.time()-started

def main():
    torch.set_num_threads(4)
    methods=[('dit','selective_dit',False),('regression','selective_regression',False),
             ('temporal_dit','temporal_selective_dit',True),('temporal_regression','temporal_selective_regression',True)]
    for name,folder,is_temporal in methods:
        while not (RUN/folder/'done.json').exists():time.sleep(20)
    # This script runs once after all checkpoint selection on P0003 has finished.
    destination=RUN/'locked_selective_temporal_test_results.json'
    if destination.exists():raise RuntimeError('Locked test already evaluated; do not silently repeat after tuning')
    data=torch.load(RUN/'coarse_cache.pt',map_location='cpu',weights_only=False)
    data['temporal']=torch.load(RUN/'temporal_context.pt',map_location='cpu',weights_only=False)
    ids=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='test']);rows=[data['rows'][i] for i in ids.tolist()]
    assert {r['subject'] for r in rows}=={'P0010','P0015'} and all(r['box_source']=='predicted' for r in rows)
    tune=torch.tensor([i for i,r in enumerate(data['rows']) if r['role']=='tune'])
    threshold=float(torch.quantile(data['confidence'][tune,1:][:,EVAL_INDICES].reshape(-1),.2))
    gt=data['gt'][ids].numpy();coarse=data['coarse'][ids].numpy();confidence=data['confidence'][ids,1:].numpy()
    valid=np.array([r['projection_valid'] for r in rows],bool);vis=np.array([r['visible_fraction'] for r in rows]);scores=np.array([r['box_score'] for r in rows])
    in_view=valid.sum(-1)>=18;raymask,raycount=ray_masks(gt,valid)
    canonical=np.zeros((len(ids),20),bool);canonical[:,EVAL_INDICES]=True
    clusters=np.array([r['sequence'] for r in rows]);groups=dict(all=canonical,
        low_pose_confidence=canonical&(confidence<threshold),
        low_box_confidence=canonical&(scores<.5)[:,None],
        high_occlusion_in_view=canonical&((vis<.5)&in_view)[:,None],
        severe_occlusion_in_view=canonical&((vis<.25)&in_view)[:,None],
        multiple_ray_aligned_fingers=canonical&raymask&((raycount>=2)&in_view)[:,None],
        low_confidence_ray_aligned=canonical&raymask&(confidence<threshold)&((raycount>=2)&in_view)[:,None])
    result=dict(test_subjects=['P0010','P0015'],tune_subjects=['P0003'],samples=len(rows),
        low_pose_confidence_threshold=threshold,ray_angle_degrees=15,
        visibility_note='Modeled whole-hand visibility with >=18 valid projected landmarks; no per-keypoint occlusion truth',
        test_frame_stride=5,coarse_training_clips=303,residual_training_clips=77,methods={},
        gate_policy='Camera-space per-point gates; strength/threshold selected on P0003 with <=5percent preservation harm',
        temporal_policy='Current plus strictly past observations; all proposals tracked before GT assignment, no GT tracks or camera-motion correction')
    preds={}
    for name,folder,is_temporal in methods:
        kind='regression' if 'regression' in name else 'dit'
        model=(TemporalResidualModel(kind=kind) if is_temporal else ResidualModel(kind=kind)).to('cuda:0');checkpoint=torch.load(RUN/folder/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint['model']);model.eval()
        pred,raw,gates,seconds=infer(model,data,ids,calibration=checkpoint['calibration']);preds[name]=pred
        full=compare(coarse,pred,gt)
        for k in ['coarse','refined']:full[k].pop('sample_mpjpe19_mm')
        per_subject={s:compare(coarse[np.array([r['subject']==s for r in rows])],pred[np.array([r['subject']==s for r in rows])],gt[np.array([r['subject']==s for r in rows])]) for s in result['test_subjects']}
        for s in per_subject:
            for k in ['coarse','refined']:per_subject[s][k].pop('sample_mpjpe19_mm')
        result['methods'][name]=dict(overall=full,groups={group:group_report(pred,gt,coarse,mask,clusters) for group,mask in groups.items()},
            ungated_groups={group:group_report(raw,gt,coarse,mask,clusters) for group,mask in groups.items()},per_subject=per_subject,
            inference_seconds=seconds,mean_gates=float(gates[:,1:][:,EVAL_INDICES].mean()),root_mean_gate=float(gates[:,0].mean()),
            selected_checkpoint_step=checkpoint['step'],calibration=checkpoint['calibration'])
        np.savez_compressed(RUN/f'test_predictions_selective_{name}.npz',coarse=coarse,gt=gt,refined=pred,ungated=raw,gates=gates,
            sequence=clusters,subject=np.array([r['subject'] for r in rows]),frame=np.array([r['frame'] for r in rows]),clip=np.array([r['clip'] for r in rows]))
        del model;torch.cuda.empty_cache()
    result['dit_vs_regression']={name:group_report(preds['dit'],gt,preds['regression'],mask,clusters) for name,mask in groups.items()}
    result['temporal_dit_vs_temporal_regression']={name:group_report(preds['temporal_dit'],gt,preds['temporal_regression'],mask,clusters) for name,mask in groups.items()}
    associations=[json.loads(l) for l in (RUN/'detector_associations.jsonl').read_text().splitlines()]
    test_assoc=[r for r in associations if r['role']=='test'];recalled=sum(r['box_iou']>=.5 for r in test_assoc)
    result['detector']=dict(eligible_gt_hands=len(test_assoc),box_recall_iou50=recalled/max(1,len(test_assoc)),
        pose_association_recall_iou30=sum(r['box_matched'] for r in test_assoc)/max(1,len(test_assoc)),
        note='All refinement methods share exactly the same predicted boxes; refinement does not change box recall')
    # Predeclared positive evidence requires error gains in hard subsets and low harm;
    # descriptive gains alone with CIs crossing zero are marked inconclusive.
    findings={}
    for method,statistics in result['methods'].items():
        all_group=statistics['groups']['all'];preserved=statistics['overall']['correct_joints_harmed_fraction']<=.05 and all_group['correct_relative_joints_harmed_fraction']<=.05
        statistics['preservation_passed']=bool(preserved)
        statistics['end_to_end_pck_camera10']=float((errors(preds[method],gt)['camera'][:,EVAL_INDICES]<=10).sum()/(len(test_assoc)*19))
        findings[method]={}
        for group,metric in [('low_pose_confidence','relative'),('high_occlusion_in_view','relative'),('multiple_ray_aligned_fingers','axial_relative')]:
            g=statistics['groups'][group];v=g.get(metric,{})
            findings[method][group]=dict(improvement_pct=v.get('improvement_pct'),ci95=v.get('ci95'),preservation_passed=bool(preserved),
                supported=bool(preserved and g['joints']>=100 and v.get('improvement_pct',0)>=5 and v.get('ci95') is not None and v['ci95'][1]<0))
    result['evidence']=findings
    part=destination.with_suffix('.json.partial');part.write_text(json.dumps(result,indent=2));part.replace(destination)
    (RUN/'evaluation_selective_done.json').write_text(json.dumps(dict(completed=True,report=str(destination))))
    print(json.dumps(dict(evidence=findings,detector=result['detector'],samples=len(rows)),indent=2),flush=True)

if __name__=='__main__':main()
