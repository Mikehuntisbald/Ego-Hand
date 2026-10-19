"""Post-scoring GT best-of-K capacity only; NEVER an inference selector."""
import json
from pathlib import Path
import numpy as np
def assess_candidate_capacity(records,GT,root):
    bank={};points=[]
    for record in records:
        clip=record['clip']
        if clip not in bank:bank[clip]=np.load(root/f'candidates/clip-{clip:06}/candidates.npz')
        points.append(bank[clip]['joints_camera21'][record['observation']])
    p=np.asarray(points);relative=p-p[:,:,:1];target=GT-GT[:,:1];error=np.linalg.norm(relative-target[:,None],axis=-1)[:,:,1:]*1000;hand=error.mean(-1);best6=hand.argmin(1);best2=hand[:,:2].argmin(1);chosen=error[np.arange(len(error)),best6];base=error[:,1];mean=error[:,0];good=base<=10;hard=base>20;spread=np.linalg.norm(p[:,2:]-p[:,:1],axis=-1)*1000
    report=dict(complete=True,posthoc_GT_only=True,deployable=False,not_used_for_inference_or_selection=True,hand_frames=len(records),reference_mean_relative20_mm=float(mean.mean()),WiLoR_relative20_mm=float(base.mean()),GT_best_of2_handframe_relative20_mm=float(hand[np.arange(len(hand)),best2].mean()),GT_best_of6_handframe_relative20_mm=float(chosen.mean()),GT_best_of6_candidate_counts=np.bincount(best6,minlength=6).tolist(),GT_best_of6_harm_over1mm=int((good&(chosen>base+1)).sum()),GT_best_of6_recovered_to20=int((hard&(chosen<=20)).sum()),original_good10=int(good.sum()),original_hard20=int(hard.sum()),protected_draw_to_mean_distance_mm=dict(mean=float(spread.mean()),P95=float(np.quantile(spread,.95))),scope='One native candidate chosen per hand frame by GT only for capacity diagnosis; no joint mixing; no temporal consistency requirement; cannot be deployed or compared as automatic output. Historical six recordings.')
    (root/'receipts/candidate_capacity_diagnostic.json').write_text(json.dumps(report,indent=2))
