"""Change only predicted handedness-derived observations; preserve RGB/labels."""
import hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,torch
from hand3d_v8_common import V7,save
from hand3d_rollout_v8 import project_fisheye624
from audit_side_consensus_v16 import votes,POLICY
from compare_detectors import iou
from offline_rgb_encoder import crop_roi
from dit_v3_inference import ObservationEncoder,prepare_observation
import spatial_rgb_common as s
RUN=V7.parent/'side_data_v16';SOURCE=V7.parent/'aligned_density_v13'

def main():
    torch.set_num_threads(4);cv2.setNumThreads(0);RUN.mkdir(exist_ok=True);started=time.time()
    data=torch.load(SOURCE/'dense_data.pt',weights_only=False,mmap=True);records=json.loads((V7.parent/'dense_sampling_v13/fresh_rows.json').read_text());consensus=votes(records);choices={}
    for fid,r in enumerate(records,1):
        # The original WiLoR side selector used its1.3 ROI, whereas the visual
        # branch/vote evidence uses1.4. Reconstruct its ORIGINAL class exactly
        # before comparing XYZ. Otherwise near overlapping hands the classes
        # can differ and invalidate the numerical-control reconstruction.
        box=np.asarray(r['box'],float);center=(box[:2]+box[2:])/2
        size=max(float((box[2:]-box[:2]).max())*1.3,24.)
        native_roi=np.r_[center-size/2,center+size/2].astype(np.float32)
        side=r['side_predictions'];overlap=iou([native_roi],side['boxes'])[0]
        if len(overlap) and overlap.max()>=POLICY['min_side_iou']:
            j=int(overlap.argmax());right=int(side['classes'][j]);confidence=float(side['scores'][j])
        else:right=1;confidence=0.
        vote=consensus.get((r['sequence'],r['clip'],r['track_id']))
        if vote and vote['eligible'] and confidence<POLICY['max_center_confidence_to_override'] and vote['right']!=right:
            choices[fid]=dict(fid=fid,image=r['image'],camera=r['camera'],box=r['box'],original_right=right,candidate_right=vote['right'],original_confidence=confidence)
    # The original center bank is appended to dense observations. Decisions for
    # these centers were already frozen using the same prediction-only votes.
    audit=json.loads((V7.parent/'side_consensus_v16/prediction_only_changed_centers.json').read_text());mapping={int(i):int(data['feature_ids'][k,8]) for k,i in enumerate(data['source_window_indices'])}
    for row in audit:
        if row['set']!='train_development':continue
        fid=mapping[row['window_index']]
        choice={k:row[k] for k in ['image','camera','box','original_right','candidate_right','original_confidence']};choice['fid']=fid
        if fid in choices:assert choices[fid]['candidate_right']==choice['candidate_right']
        choices[fid]=choice
    rows=list(choices.values());save(RUN/'prediction_only_changes.json',rows)
    def prep(row):return prepare_observation(cv2.imread(row['image']),row['box'],row['camera'],dict(boxes=[],classes=[],scores=[]))
    with ThreadPoolExecutor(max_workers=8) as pool:prepared=list(pool.map(prep,rows))
    encoder=ObservationEncoder('cuda:3');out={}
    for label in ['original','candidate']:
        chunks=[]
        for start in range(0,len(rows),16):
            part=prepared[start:start+16];rr=rows[start:start+16];n=len(part);part=part+[part[-1]]*(16-n);rr=rr+[rr[-1]]*(16-n)
            p=encoder.features([x['wilor_crop'] for x in part],[x['rotation'] for x in part],[x['focal'] for x in part],[x[f'{label}_right'] for x in rr],[x['original_confidence'] for x in rr],[0.]*16,[x['reprojection_error'] for x in part],[x['crop_fallback'] for x in part])
            chunks.append(p['wilor'][:n].cpu())
            if start%128==0:print(json.dumps(dict(stage=label,done=start+n,total=len(rows),seconds=time.time()-started)),flush=True)
        out[label]=torch.cat(chunks)
    fids=torch.tensor([r['fid'] for r in rows]);difference=(out['original']-data['xyz_camera_bank'][fids]).norm(dim=-1)*1000
    parity=dict(max_mm=float(difference.max()),mean_mm=float(difference.mean()));save(RUN/'reconstruction_parity.json',parity);assert difference.max()<.25,parity
    candidate=data['xyz_camera_bank'].clone();candidate[fids]=out['candidate'];params=torch.load(SOURCE/'camera_params.pt',weights_only=False,mmap=True)
    for variant,xyz in [('control',data['xyz_camera_bank']),('consensus',candidate)]:
        dest=RUN/variant;dest.mkdir(exist_ok=True);new={**data};new['xyz_camera_bank']=xyz
        for name in ['native_bank.pt','trajectory_targets.pt']:
            link=dest/name
            if not link.exists():link.symlink_to(SOURCE/name)
        new['world']=torch.einsum('njc,nkc->njk',xyz,data['rotation'])+data['translation'][:,None]
        uv=project_fisheye624(xyz,params)/1408;observed=torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<1).all(-1)&data['available_bank']
        # Both arms use exactly the same physical XYZ->XY adapter. Its origin
        # differs from the historical center2D cache and is declared explicitly.
        new['xy']=torch.nan_to_num(uv)[data['feature_ids']];new['observed_2d']=observed[data['feature_ids']]
        new['original_base_for_evaluation']=data['xyz_camera_bank'][data['feature_ids'][:,8]].clone()
        assert torch.equal(new['gt'],data['gt']) and torch.equal(new['valid'],data['valid'])
        torch.save(new,dest/'dense_data.pt')
        save(dest/'ready.json',dict(complete=True,windows=len(data['roles']),roles={r:data['roles'].count(r) for r in set(data['roles'])},original_center_rgb_and_targets_exact=True,variant=variant,physical_xy_shared_contract=True,native_bank=str(SOURCE/'native_bank.pt'),trajectory_targets=str(SOURCE/'trajectory_targets.pt'),policy=POLICY,scope='Original3417train/development centers only. Prediction-only votes; no artificial occlusion; no fourth batch/retained failures included. Upstream XYZ/derivedXY changed, re-fit risks before3D training.'))
    save(RUN/'ready.json',dict(complete=True,changed_observations=len(rows),parity=parity,source_data_sha256=hashlib.sha256((SOURCE/'dense_data.pt').read_bytes()).hexdigest(),source_manifest=str(V7.parent/'dense_sampling_v13/fresh_manifest.json'),scope='Existingtrain/development only; nativeRGB/3Dlabels stay unchanged; control/consensus differ only in predicted handedness-derivedXYZ and derivedXY',seconds=time.time()-started))
    print((RUN/'ready.json').read_text(),flush=True)

if __name__=='__main__':main()
