"""Reconstruct changed-side centers before deciding whether to retrain DiT."""
import json
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,torch
from hand3d_v8_common import V7,load,save,metrics
from dit_v3_inference import ObservationEncoder,prepare_observation
import spatial_rgb_common as s
RUN=V7.parent/'side_consensus_v16'

@torch.inference_mode()
def main():
    torch.set_num_threads(4);cv2.setNumThreads(0)
    audit=json.loads((RUN/'audit_results.json').read_text());data=load('cpu');records,_=s.records_and_index();changed=[]
    for label,value in audit['sets'].items():
        for row in value['rows']:
            if row['changed']:
                r=records[int(data['feature_ids'][row['window_index'],8])-1]
                # No GT pose/side/visibility is included in reconstruction input.
                changed.append(dict(set=label,window_index=row['window_index'],role=row['role'],image=r['image'],camera=r['camera'],box=r['box'],original_right=row['original_right'],candidate_right=row['candidate_right'],original_confidence=row['original_confidence'],consensus=row['consensus']))
    save(RUN/'prediction_only_changed_centers.json',changed)
    def prep(row):return prepare_observation(cv2.imread(row['image']),row['box'],row['camera'],dict(boxes=[],classes=[],scores=[]))
    with ThreadPoolExecutor(max_workers=8) as pool:prepared=list(pool.map(prep,changed))
    encoder=ObservationEncoder('cuda:3');outputs={}
    for label in ['original','candidate']:
        xyz=[]
        for start in range(0,len(changed),16):
            part=prepared[start:start+16];rr=changed[start:start+16];n=len(part);part=part+[part[-1]]*(16-n);rr=rr+[rr[-1]]*(16-n)
            result=encoder.features([p['wilor_crop'] for p in part],[p['rotation'] for p in part],[p['focal'] for p in part],[r[f'{label}_right'] for r in rr],[r['original_confidence'] for r in rr],[0.]*16,[p['reprojection_error'] for p in part],[p['crop_fallback'] for p in part])
            xyz.append(result['wilor'][:n].cpu())
        outputs[label]=torch.cat(xyz)
    indices=torch.tensor([r['window_index'] for r in changed]);base=data['xyz_camera_bank'][data['feature_ids'][indices,8]]
    difference=(outputs['original']-base).norm(dim=-1)*1000;parity=dict(max_mm=float(difference.max()),mean_mm=float(difference.mean()))
    save(RUN/'reconstruction_parity.json',parity);assert float(difference.max())<.25,parity
    outcomes={};details=[]
    for label,value in audit['sets'].items():
        for role in sorted({r['role'] for r in value['rows']}):
            selected=[r['window_index'] for r in value['rows'] if r['role']==role];ids=torch.tensor(selected);old=data['xyz_camera_bank'][data['feature_ids'][ids,8]];pred=old.clone();lookup={ix:i for i,ix in enumerate(selected)}
            for j,r in enumerate(changed):
                if r['set']==label and r['role']==role:pred[lookup[r['window_index']]]=outputs['candidate'][j]
            outcomes[f'{label}_{role}']=dict(windows=len(ids),baseline=metrics(old,old,data['gt'][ids],data['valid'][ids]),candidate=metrics(pred,old,data['gt'][ids],data['valid'][ids]))
    for j,r in enumerate(changed):
        ix=r['window_index'];detail={k:r[k] for k in ['set','window_index','role','original_right','candidate_right']}
        detail.update(baseline=metrics(base[j:j+1],base[j:j+1],data['gt'][ix:ix+1],data['valid'][ix:ix+1]),candidate=metrics(outputs['candidate'][j:j+1],base[j:j+1],data['gt'][ix:ix+1],data['valid'][ix:ix+1]));details.append(detail)
    torch.save(dict(indices=indices,baseline=base,original_recomputed=outputs['original'],candidate=outputs['candidate']),RUN/'changed_centers_xyz.pt')
    save(RUN/'center_xyz_results.json',dict(complete=True,parity=parity,groups=outcomes,cases=details,scope='Prediction-only temporal votes and reconstructed XYZ; GT used only after inference. Training/development and retained-failure diagnostics; no independent adoption evidence.'))
    print(json.dumps(dict(complete=True,parity=parity,groups=outcomes),indent=2),flush=True)

if __name__=='__main__':main()
