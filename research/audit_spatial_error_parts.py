import json
import spatial_rgb_common as s
import numpy as np
O=s.RUN/'hard_case_audit';d=np.load(O/'case_metrics.npz');p=np.load(s.RUN/'rgb_dit_predictions.npz')
gt=p['gt'];xy=p['predicted'];m=p['mask'];counts=m.sum(1).clip(1)
wrist=np.linalg.norm(xy[:,5]-gt[:,5],axis=-1)*1408
relative=np.linalg.norm((xy-xy[:,5:6])-(gt-gt[:,5:6]),axis=-1)*1408
rel=(relative*m).sum(1)/counts
offset=((xy-gt)*m[...,None]).sum(1)/counts[:,None]
centered=(np.linalg.norm(xy-gt-offset[:,None],axis=-1)*1408*m).sum(1)/counts
result={}
for mode in ['partial','full','natural']:
    sel=(d['mode']==mode)&(d['count']>=3);failed=sel&d['failed'];rec=sel&d['recovered']
    result[mode]={}
    for name,q in [('all',sel),('failed',failed),('recovered',rec)]:
        if not q.any():continue
        result[mode][name]=dict(cases=int(q.sum()),mean_absolute_error_px=float(d['dit_px'][q].mean()),mean_wrist_error_px=float(wrist[q].mean()),
            mean_wrist_aligned_finger_error_px=float(rel[q].mean()),fraction_wrist_aligned_le20=float((rel[q]<=20).mean()),
            mean_centroid_aligned_error_px=float(centered[q].mean()),centroid_aligned_le20_fraction=float((centered[q]<=20).mean()),centroid_aligned_gt30_fraction=float((centered[q]>30).mean()),
            fraction_wrist_aligned_gt40=float((rel[q]>40).mean()),median_error_over_roi=float(np.median(d['dit_px'][q]/d['roi_size_px'][q])))
    result[mode]['tips']={finger:dict(joints=int((m[:,j]&sel).sum()),mean_px=float(np.linalg.norm(xy[:,j]-gt[:,j],axis=-1)[m[:,j]&sel].mean()*1408)) for j,finger in enumerate(['thumb','index','middle','ring','little'])}
    result[mode]['by_sequence']={str(seq):dict(cases=int((sel&(d['sequence']==seq)).sum()),failed=int((failed&(d['sequence']==seq)).sum()),recovered=int((rec&(d['sequence']==seq)).sum())) for seq in np.unique(d['sequence'])}
result['scope']='Post-hoc GT wrist alignment for error diagnosis only; no GT alignment used in inference; not a causal decomposition.'
s.save(O/'error_parts.json',result);print(json.dumps(result,indent=2))
