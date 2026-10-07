"""One final evaluation after all v2 model/calibration choices are frozen."""
import json,time
import numpy as np,torch
from residual_models import ResidualModel
from metrics_3d import EVAL_INDICES,compare
from evaluate_proof import ray_masks,group_report,errors
from oof_calibration import active_data,frozen_gates,raw_sigma
from oof_common import OLD,RUN,save,sha

def infer(model,data,ids,checkpoint,v1=False):
    outputs=[];props=[];logits=[];gates=[];generator=torch.Generator(device='cuda:1').manual_seed(901003)
    with torch.inference_mode():
        for start in range(0,len(ids),256):
            ix=ids[start:start+256];c=data['coarse'][ix].to('cuda:1');f=data['confidence'][ix].to('cuda:1');rgb=data['rgb'][ix].to('cuda:1').float()
            with torch.autocast('cuda',dtype=torch.bfloat16):
                p=model.propose(c,f,rgb,steps=10,samples=4 if model.kind=='dit' else 1,generator=generator)
                l,g=model.gates(p,c,f,rgb)
                if v1:g=g*(g>=checkpoint['calibration']['threshold'])*checkpoint['calibration']['strength']
                else:g=frozen_gates(model,l,checkpoint['calibration'])
                pred=model.apply_gates(p,c,g)
            outputs.append(pred.float().cpu());props.append(p.float().cpu());logits.append(l.float().cpu());gates.append(g.float().cpu())
    return torch.cat(outputs).numpy(),torch.cat(props),torch.cat(logits),torch.cat(gates)

def evaluate_subset(raw,role,checkpoints,threshold):
    ids=torch.tensor([i for i,r in enumerate(raw['rows']) if r['role']==role]);rows=[raw['rows'][i] for i in ids.tolist()]
    assert {r['subject'] for r in rows}=={'P0010','P0015'} and all(r['box_source']=='predicted' for r in rows)
    gt=raw['gt'][ids].numpy();coarse=raw['in_subject_coarse'][ids].numpy();conf=raw['in_subject_confidence'][ids,1:].numpy()
    valid=np.array([r['projection_valid'] for r in rows],bool);vis=np.array([r['visible_fraction'] for r in rows]);scores=np.array([r['box_score'] for r in rows])
    raymask,raycount=ray_masks(gt,valid);inview=valid.sum(-1)>=18
    canonical=np.zeros((len(ids),20),bool);canonical[:,EVAL_INDICES]=True
    clusters=np.array([r['sequence'] for r in rows]);groups=dict(all=canonical,
        low_pose_confidence=canonical&(conf<threshold),low_box_confidence=canonical&(scores<.5)[:,None],
        high_occlusion_in_view=canonical&((vis<.5)&inview)[:,None],
        severe_occlusion_in_view=canonical&((vis<.25)&inview)[:,None],
        multiple_ray_aligned_fingers=canonical&raymask&((raycount>=2)&inview)[:,None])
    result=dict(samples=len(ids),subjects=['P0010','P0015'],source_sequences=sorted(set(clusters)),methods={},evidence={})
    predictions={}
    for method,ck in checkpoints.items():
        if method=='v1_dit':
            data=active_data(raw,'in_subject_dit');data['rgb']=raw['v1_rgb']
        else:data=active_data(raw,method,ck['uncertainty_calibration'])
        kind='regression' if method=='oof_regression' else 'dit'
        model=ResidualModel(kind=kind).to('cuda:1');model.load_state_dict(ck['model']);model.eval()
        torch.cuda.synchronize();started=time.perf_counter()
        pred,proposal,logits,gates=infer(model,data,ids,ck,v1=method=='v1_dit');predictions[method]=pred
        torch.cuda.synchronize();seconds=time.perf_counter()-started
        overall=compare(coarse,pred,gt)
        for k in ['coarse','refined']:overall[k].pop('sample_mpjpe19_mm')
        stats=dict(overall=overall,groups={k:group_report(pred,gt,coarse,m,clusters) for k,m in groups.items()},
            mean_applied_gate=float(gates[:,1:][:,EVAL_INDICES].mean()),calibration=ck['calibration'],
            refinement_seconds=seconds,refinement_ms_per_hand=seconds*1000/len(ids),
            timing_scope='Batched residual + gate inference and host transfer only; detector, common RGB and coarse backbone excluded')
        stats['preservation_passed']=bool(overall['correct_joints_harmed_fraction']<=.05 and stats['groups']['all']['correct_relative_joints_harmed_fraction']<=.05)
        stats['per_subject']={}
        for subject in ['P0010','P0015']:
            m=np.array([r['subject']==subject for r in rows]);s=compare(coarse[m],pred[m],gt[m])
            for k in ['coarse','refined']:s[k].pop('sample_mpjpe19_mm')
            stats['per_subject'][subject]=s
        sigma=raw_sigma(data['confidence'][ids]).numpy()
        relative=((coarse-coarse[:,5:6])-(gt-gt[:,5:6]))
        stats['uncertainty']=dict(predicted_relative_sigma_mm=float(sigma[:,1:][:,EVAL_INDICES].mean()*1000),
            actual_relative_axis_rms_mm=float(np.sqrt((relative[:,EVAL_INDICES]**2).mean())*1000))
        if method!='v1_dit':
            raw_full=coarse+proposal[:,:1].numpy()*.1+proposal[:,1:].numpy()*.03
            useful=(np.linalg.norm(raw_full-gt,axis=-1)+.001<np.linalg.norm(coarse-gt,axis=-1)).astype(float)
            target=np.concatenate([useful[:,5:6],useful],1);p=ck['calibration']['probability'];prob=(p['a']*logits+p['b']).sigmoid().numpy()
            mask=np.ones_like(prob,bool);mask[:,6]=False
            stats['gate_probability_brier']=float(((prob-target)**2)[mask].mean())
        result['methods'][method]=stats;result['evidence'][method]={}
        for group,metric in [('low_pose_confidence','relative'),('high_occlusion_in_view','relative'),('multiple_ray_aligned_fingers','axial_relative')]:
            g=stats['groups'][group];v=g.get(metric,{})
            result['evidence'][method][group]=dict(improvement_pct=v.get('improvement_pct'),ci95=v.get('ci95'),
                preservation_passed=stats['preservation_passed'],supported=bool(stats['preservation_passed'] and g['joints']>=100 and v.get('improvement_pct',0)>=5 and v.get('ci95') is not None and v['ci95'][1]<0))
        np.savez_compressed(RUN/f'{role}_{method}_predictions.npz',coarse=coarse,gt=gt,refined=pred,proposal=proposal.numpy(),logits=logits.numpy(),gates=gates.numpy(),sequence=clusters,subject=np.array([r['subject'] for r in rows]))
        del model;torch.cuda.empty_cache()
    result['oof_vs_in_subject_dit']={k:group_report(predictions['oof_dit'],gt,predictions['in_subject_dit'],m,clusters) for k,m in groups.items()}
    result['oof_dit_vs_regression']={k:group_report(predictions['oof_dit'],gt,predictions['oof_regression'],m,clusters) for k,m in groups.items()}
    if role=='fresh':
        a=json.loads((RUN/'fresh_associations.json').read_text());result['detector']=dict(eligible_hands=len(a),recall_iou50=sum(r['iou']>=.5 for r in a)/len(a),association_iou30=sum(r['matched'] for r in a)/len(a))
    return result

def main():
    torch.set_num_threads(4);dest=RUN/'final_results.json'
    if dest.exists():raise RuntimeError('Final data already evaluated; no silent retuning/re-evaluation')
    protocol=json.loads((RUN/'protocol.json').read_text());checkpoints={};lock=[]
    for method in protocol['methods']:
        assert (RUN/method/'done.json').exists()
        path=RUN/method/'best.pt';checkpoints[method]=torch.load(path,map_location='cpu',weights_only=False)
        lock.append(dict(method=method,path=str(path),sha256=sha(path)))
    checkpoints['v1_dit']=torch.load(OLD/'selective_dit/best.pt',map_location='cpu',weights_only=False)
    save(RUN/'evaluation_lock.json',dict(protocol_sha256=sha(RUN/'protocol.json'),checkpoints=lock,fresh_results_seen_before_lock=False))
    raw=torch.load(RUN/'oof_cache.pt',map_location='cpu',weights_only=False)
    tune=torch.tensor([i for i,r in enumerate(raw['rows']) if r['subject']=='P0003'])
    threshold=float(torch.quantile(raw['in_subject_confidence'][tune,1:][:,EVAL_INDICES].reshape(-1),.2))
    result=dict(study=protocol['study'],protocol=protocol,raw_low_confidence_threshold=threshold,
        limitations=['Fresh source sequences, reused subjects; no claim of entirely new-subject study',
            'Whole-hand modeled visibility only; ray alignment is a proxy',
            'Conditional 3D refinement cannot improve detector box recall',
            'Calibration Wilson interval is descriptive under dependent joints; not a distribution-free safety guarantee'])
    # Both sets evaluated under the same pre-locked checkpoints, never used for selection.
    result['development']=evaluate_subset(raw,'development',checkpoints,threshold)
    result['fresh']=evaluate_subset(raw,'fresh',checkpoints,threshold)
    save(dest,result);save(RUN/'evaluation_done.json',dict(complete=True,results=str(dest)))
    print(json.dumps(dict(fresh=result['fresh']['evidence'],development=result['development']['evidence']),indent=2),flush=True)
if __name__=='__main__':main()
