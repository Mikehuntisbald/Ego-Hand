"""Post-freeze paired training-path effects and fast finger-motion retention."""
import json,sys
from pathlib import Path
import numpy as np
V=Path('/mnt/why/native_mano_multiseq_v3_20261019');R=Path('/mnt/why/native_mano_rollout_v4_20261019')
sys.path.insert(0,str(V/'src'))
from evaluate_validation import clustered_ci,stats,MAPPING,WRIST

def main():
    d=json.loads((V/'evaluation_r3/metrics.json').read_text());protocol=json.loads((V/'protocol.json').read_text());pairs={}
    for new,old in [('rollout_dit','cosine_dit'),('rollout_protected','cosine_protected'),('rollout_dit_displacement','cosine_protected_displacement')]:
        rows=d['per_clip'];values=[dict(n=r['metrics'][new]['relative_without_wrist_mm']['n'],sum=r['metrics'][new]['relative_without_wrist_mm']['n']*(r['metrics'][new]['relative_without_wrist_mm']['mean']-r['metrics'][old]['relative_without_wrist_mm']['mean'])) for r in rows]
        pairs[f'{new}_minus_{old}']=dict(recording=clustered_ci(values,[r['sequence'] for r in rows]),subject=clustered_ci(values,[r['subject'] for r in rows]))
        for interval in pairs[f'{new}_minus_{old}'].values():interval['units']=f'{new} minus {old} mm; negative is better'
    arms=['WiLoR','regression','cosine_dit','cosine_protected_displacement','rollout_dit','rollout_protected','rollout_dit_displacement'];motion={a:dict(pred=[],GT=[],error=[]) for a in arms}
    labels=np.load(V/'evaluation_r3/labels_and_errors.npz')
    for clip in protocol['clips']:
        view=Path(clip['view']);rows=json.loads((view/'outputs/wilor_clip2102/observations.json').read_text())['frames'];poses=json.loads((view/'outputs/head_clip2102/headpose.json').read_text())['frames']
        GT=labels[f'clip{clip["clip"]}_GT_camera20'];valid=labels[f'clip{clip["clip"]}_valid'];rotation=np.array([poses[r['frame_index']]['T_world_from_rgb'] for r in rows])[:,:3,:3]
        rel=lambda x:x-x[:,WRIST:WRIST+1]
        gt=np.einsum('nij,nkj->nki',rotation,rel(GT));pred={a:np.load(view/('predictions_rollout' if a.startswith('rollout_') else 'predictions')/f'{a}.npz')['joints_camera21'][:,MAPPING] for a in arms}
        for tid in sorted({r['track_id'] for r in rows}):
            ids=np.array([i for i,r in enumerate(rows) if r['track_id']==tid and valid[i].all()]);ids=ids[np.argsort([rows[i]['timestamp_ns'] for i in ids])]
            if len(ids)<2:continue
            dt=np.diff([rows[i]['timestamp_ns'] for i in ids])/1e9;pair=(dt>0)&(dt<.1)
            if not pair.any():continue
            vg=np.diff(gt[ids],axis=0)/(60*dt[:,None,None]);speed=np.linalg.norm(vg,axis=-1);mask=np.zeros_like(speed,dtype=bool);mask[:,np.arange(20)!=WRIST]=True;mask&=pair[:,None]
            fast=mask&(speed>=np.quantile(speed[mask],.9))
            for a,x in pred.items():
                world=np.einsum('nij,nkj->nki',rotation[ids],rel(x[ids]));vp=np.diff(world,axis=0)/(60*dt[:,None,None]);motion[a]['pred'].extend(np.linalg.norm(vp,axis=-1)[fast]);motion[a]['GT'].extend(speed[fast]);motion[a]['error'].extend((np.linalg.norm(vp-vg,axis=-1)*1000)[fast])
    result=dict(complete=True,posthoc_diagnostic_only=True,changes_no_predictions_or_admission=True,paired_training_path_effect=pairs,
        fast_relative_hand_motion={a:dict(n=len(m['pred']),amplitude_ratio=float(np.mean(m['pred'])/max(np.mean(m['GT']),1e-12)),error_mm_at60Hz=stats(m['error'])) for a,m in motion.items()},
        definition='relative to wrist, aligned through supplied camera rotation to world; top decile of GT motion per predicted track, exclude zero wrist; fixed predictions before these GT summaries')
    (R/'receipts/paired_diagnostics_r1.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
