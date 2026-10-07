"""Predict handedness/XYZ on fifth observations without computing GT metrics."""
import json,time
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,torch
from hand3d_v8_common import V7,save
from hand3d_rollout_v8 import project_fisheye624
from audit_side_consensus_v16 import votes,POLICY
from compare_detectors import iou
from dit_v3_inference import ObservationEncoder,prepare_observation
import spatial_rgb_common as s
RUN=V7.parent/'fifth_dense_v16'

@torch.inference_mode()
def main():
    torch.set_num_threads(4);cv2.setNumThreads(0);assert json.loads((RUN/'prepared.json').read_text())['complete']
    assert not (RUN/'evaluation_results.json').exists()
    data=torch.load(RUN/'dense_data.pt',weights_only=False,mmap=True);records=json.loads((RUN/'fresh_rows.json').read_text());consensus=votes(records);choices=[];params=torch.zeros(len(records)+1,16)
    for fid,r in enumerate(records,1):
        cam=s.common.from_json(r['camera']);params[fid]=torch.tensor(list(cam.f)+list(cam.c)+list(cam.distort))
        box=np.asarray(r['box'],float);center=(box[:2]+box[2:])/2;size=max(float((box[2:]-box[:2]).max())*1.3,24.);roi=np.r_[center-size/2,center+size/2].astype(np.float32)
        side=r['side_predictions'];overlap=iou([roi],side['boxes'])[0]
        if len(overlap) and overlap.max()>=POLICY['min_side_iou']:
            j=int(overlap.argmax());right=int(side['classes'][j]);confidence=float(side['scores'][j])
        else:right=1;confidence=0.
        vote=consensus.get((r['sequence'],r['clip'],r['track_id']))
        if vote and vote['eligible'] and confidence<POLICY['max_center_confidence_to_override'] and vote['right']!=right:
            choices.append(dict(fid=fid,image=r['image'],camera=r['camera'],box=r['box'],original_right=right,candidate_right=vote['right'],original_confidence=confidence))
    save(RUN/'side_choices_prediction_only.json',choices);xyz=data['xyz_camera_bank'].clone();parity=dict(max_mm=0.,mean_mm=0.)
    if choices:
        def prep(r):return prepare_observation(cv2.imread(r['image']),r['box'],r['camera'],dict(boxes=[],classes=[],scores=[]))
        with ThreadPoolExecutor(max_workers=8) as pool:prepared=list(pool.map(prep,choices))
        encoder=ObservationEncoder('cuda:3');out={}
        for label in ['original','candidate']:
            chunks=[]
            for start in range(0,len(choices),16):
                pp=prepared[start:start+16];rr=choices[start:start+16];n=len(pp);pp=pp+[pp[-1]]*(16-n);rr=rr+[rr[-1]]*(16-n)
                p=encoder.features([x['wilor_crop'] for x in pp],[x['rotation'] for x in pp],[x['focal'] for x in pp],[x[f'{label}_right'] for x in rr],[x['original_confidence'] for x in rr],[0.]*16,[x['reprojection_error'] for x in pp],[x['crop_fallback'] for x in pp]);chunks.append(p['wilor'][:n].cpu())
            out[label]=torch.cat(chunks)
        ids=torch.tensor([r['fid'] for r in choices]);difference=(out['original']-xyz[ids]).norm(dim=-1)*1000;parity=dict(max_mm=float(difference.max()),mean_mm=float(difference.mean()));assert difference.max()<.25,parity
        xyz[ids]=out['candidate']
    uv=project_fisheye624(xyz,params)/1408;observed=torch.isfinite(uv).all(-1)&(uv>=0).all(-1)&(uv<1).all(-1)&data['available_bank'];new={**data,'xyz_camera_bank':xyz,'world':torch.einsum('njc,nkc->njk',xyz,data['rotation'])+data['translation'][:,None],'xy':torch.nan_to_num(uv)[data['feature_ids']],'observed_2d':observed[data['feature_ids']]}
    assert torch.equal(new['gt'],data['gt']) and torch.equal(new['valid'],data['valid']);torch.save(new,RUN/'consensus_data.pt')
    save(RUN/'side_ready.json',dict(complete=True,changed_observations=len(choices),parity=parity,metrics_opened=False,side_policy=POLICY,scope='Fifth unused clips; votes/XYZ use predictions and RGB only; GT payload preserved but neverreadbychoices/reconstruction. No model metric computed.'))
    print((RUN/'side_ready.json').read_text(),flush=True)

if __name__=='__main__':main()
