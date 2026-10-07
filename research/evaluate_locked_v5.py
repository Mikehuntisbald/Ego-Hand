"""One sealed evaluation; do not reuse these labels for further selection."""
import hashlib
import json
import sys
import os
import time
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D');RUN=Path(os.environ.get('HOT3D_DIT_RUN',str(ROOT/'experiments/dit_wilor_v5')))
sys.path.insert(0,str(RUN/'sealed'))
import wilor_eval_common
import numpy as np
import torch
from dit_v3_inference import Refiner
from metrics_3d import EVAL_INDICES,compare
from evaluate_proof import ray_masks,group_report,errors


def save(path,obj):
    temp=path.with_suffix('.partial');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)


def main():
    torch.set_num_threads(4)
    while not (RUN/'locked_observations_done.json').exists():time.sleep(10)
    dest=RUN/'locked_results.json'
    if dest.exists():print('Sealed evaluation already exists; no rerun',flush=True);return
    seal=json.loads((RUN/'sealed/selection.json').read_text())
    for name,r in seal['models'].items():assert hashlib.sha256(Path(r['checkpoint']).read_bytes()).hexdigest()==r['sha256']
    rows=json.loads((RUN/'locked_rows.json').read_text());pieces={};previous=0
    for path in sorted((RUN/'locked_observations').glob('*.pt')):
        chunk=torch.load(path,map_location='cpu',weights_only=False)
        assert chunk.pop('start')==previous;previous=chunk.pop('end')
        for k,v in chunk.items():pieces.setdefault(k,[]).append(v)
    assert previous==len(rows)
    data={k:torch.cat(v).to('cuda:1') for k,v in pieces.items()};del pieces
    ids=np.array([i for i,r in enumerate(rows) if r['matched']])
    gt=np.array([rows[i]['gt'] for i in ids],np.float32);coarse=data['coarse'][ids].cpu().numpy()
    valid=np.array([rows[i]['projection_valid'] for i in ids],bool);vis=np.array([rows[i]['visibility_label'] for i in ids])
    ray,counts=ray_masks(gt,valid);canonical=np.zeros((len(ids),20),bool);canonical[:,EVAL_INDICES]=True
    confidence=data['confidence'][ids,1:].cpu().numpy();clusters=np.array([rows[i]['sequence'] for i in ids])
    groups=dict(all=canonical,low_pose_confidence=canonical&(confidence<.69746333360672),
        high_occlusion_in_view=canonical&((vis<.5)&(valid.sum(1)>=18))[:,None],
        multiple_ray_aligned_fingers=canonical&ray&((counts>=2)&(valid.sum(1)>=18))[:,None])
    counts=json.loads((RUN/'locked_detection_counts.json').read_text());results={};predictions={}
    for kind in ['dit','regression']:
        model=Refiner(RUN/'sealed'/f'{kind}.pt',device='cuda:1')
        generator=torch.Generator(device='cuda:1').manual_seed(916003);out=[];gates=[]
        started=time.time()
        for start in range(0,len(rows),64):
            b={k:v[start:start+64] for k,v in data.items()}
            result=model(b,generator=generator)
            out.append(result['xyz'].cpu());gates.append(result['gates'].cpu())
        all_predictions=torch.cat(out).numpy();all_gates=torch.cat(gates).numpy();p=all_predictions[ids]
        predictions[kind]=p
        full=compare(coarse,p,gt)
        for key in ['coarse','refined']:full[key].pop('sample_mpjpe19_mm')
        reports={name:group_report(p,gt,coarse,mask,clusters) for name,mask in groups.items()}
        for report in reports.values():
            for v in report.values():
                if isinstance(v,dict) and 'unit' in v:v['unit']='source-sequence bootstrap; mixed known and held-out training subjects, new sequences'
        protection=full['correct_joints_harmed_fraction']<=.05 and reports['all']['correct_relative_joints_harmed_fraction']<=.05
        evidence={}
        for name in ['low_pose_confidence','high_occlusion_in_view','multiple_ray_aligned_fingers']:
            r=reports[name];metric='axial_relative' if name=='multiple_ray_aligned_fingers' else 'relative';m=r.get(metric,{})
            evidence[name]=dict(joints=r['joints'],improvement_pct=m.get('improvement_pct'),ci95_change_mm=m.get('ci95'),
                passed=bool(r['joints']>=100 and m.get('improvement_pct',0)>=5 and m.get('ci95') is not None and m['ci95'][1]<0))
        camera_pass=full['refined']['mpjpe19_mm']<=full['coarse']['mpjpe19_mm']+.1
        accepted=protection and camera_pass and all(r['passed'] for r in evidence.values())
        per_subject={}
        subject_labels=np.array([rows[i]['subject'] for i in ids])
        for subject in sorted(set(subject_labels)):
            mask=subject_labels==subject
            sr=compare(coarse[mask],p[mask],gt[mask])
            for key in ['coarse','refined']:sr[key].pop('sample_mpjpe19_mm')
            per_subject[subject]=sr
        held=per_subject.get('P0015')
        held_no_regression=held is not None and held['refined']['wrist_relative_mpjpe19_mm']<=held['coarse']['wrist_relative_mpjpe19_mm']
        accepted=accepted and held_no_regression
        e=errors(p,gt)
        results[kind]=dict(overall=full,groups=reports,evidence=evidence,protection_passed=protection,camera_passed=camera_pass,accepted=accepted,per_subject=per_subject,held_subject_no_regression=held_no_regression,
            end_to_end_pck_camera10=float((e['camera'][:,EVAL_INDICES]<=10).sum()/(counts['eligible_hands']*19)),
            end_to_end_pck_relative10=float((e['relative'][:,EVAL_INDICES]<=10).sum()/(counts['eligible_hands']*19)),
            mean_gate=float(all_gates[ids].mean()),refinement_seconds=time.time()-started)
        np.savez_compressed(RUN/f'locked_{kind}_predictions.npz',coarse=coarse,gt=gt,predicted=p,gates=all_gates[ids],sample_indices=ids)
        save(RUN/f'locked_{kind}_result.json',results[kind])
        print(json.dumps(dict(kind=kind,accepted=accepted,overall=full,evidence=evidence,protection=protection)),flush=True)
        del model;torch.cuda.empty_cache()
    comparison={name:group_report(predictions['dit'],gt,predictions['regression'],mask,clusters) for name,mask in groups.items()}
    result=dict(complete=True,accepted=results['dit']['accepted'],sampled_frames=counts['frames'],matched_hands=len(ids),detection=counts,
                source_sequences=sorted(set(clusters)),subjects=sorted({rows[i]['subject'] for i in ids}),methods=results,dit_vs_regression=comparison,
                frozen_selection_sha256=hashlib.sha256((RUN/'sealed/selection.json').read_bytes()).hexdigest(),
                scope=json.loads((RUN/'protocol.json').read_text())['test_scope'],seed=916003)
    save(dest,result)
    save(RUN/'status.json',dict(stage='locked_evaluated',complete=False,acceptance_passed=result['accepted'],delivery_complete=False))
    print(json.dumps(dict(locked_evaluation_complete=True,accepted=result['accepted'])),flush=True)


if __name__=='__main__':main()

