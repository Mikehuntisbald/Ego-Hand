"""Score actual RF-DETR -> same full 3D core, after sealed output."""
import json,collections,argparse
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from build_surgical_pose_cache_v51 import MAPPING
from benchmark_rfdetr_v52 import PAIRED,box_iou,assignments
ROOT=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007')

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',default='glove_complete');p.add_argument('--name',default='full_glove');args=p.parse_args()
    folder=ROOT/args.folder;assert (folder/'freeze.json').exists()
    prediction=json.loads((folder/'prediction.json').read_text());source=json.loads((folder/'input_tracks.json').read_text());frames=json.loads((ROOT/'glove_rgb.json').read_text())['frames']
    targets={r['id']:r for r in map(json.loads,Path('/mnt/why/HOT3D/domain_data_v51/surgical_hands/selected_records.jsonl').read_text().splitlines())}
    baseline=Path('/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007/complete_glove_test');prior={r['id']:r for r in json.loads((baseline/'per_image.json').read_text())};result=[]
    for i,f in enumerate(frames):
        row=targets[f['id']];offset=np.asarray(f['image_transform']['offset']);scale=f['image_transform']['scale'];boxes=np.array([h['bbox'] for h in row['hands']])*scale+np.tile(offset,2);kp=np.array([np.asarray(h['keypoints']).reshape(21,3)[MAPPING] for h in row['hands']]);xy=kp[:,:,:2]*scale+offset;valid=kp[:,:,2]>0;visible=kp[:,:,2]==2;denom=np.mean(boxes[:,2:]-boxes[:,:2],-1)
        indices=[k for k,tr in enumerate(source['tracks']) if tr['id'].startswith(str(i)+'_')];b=np.array([source['tracks'][k]['frames'][0]['box_xyxy'] for k in indices]).reshape(-1,4);ov=box_iou(boxes,b);x,y=assignments(ov);success=np.zeros_like(valid);matched=0;errors=np.full(valid.shape,1e3)
        for gi,pj in zip(x,y):
            if ov[gi,pj]<.5:continue
            matched+=1;v=prediction['tracks'][indices[pj]]['frames'][0];xyz=np.asarray(v['candidate_xyz_camera_m']);fx,fy,cx,cy=f['camera']['calibration']['projection_params'];uv=xyz[:,:2]/xyz[:,2:]*[fx,fy]+[cx,cy];errors[gi]=np.linalg.norm(uv-xy[gi],axis=-1)/denom[gi];success[gi]=(errors[gi]<=.1)&valid[gi]
        before=np.array(prior[f['id']]['candidate']['success']);assert before.shape==success.shape
        result.append(dict(id=f['id'],group=f['group'],hands=len(boxes),matched_hands=matched,points=int(valid.sum()),RF_correct=int(success.sum()),SAM_correct=int(before.sum()),previous_correct_lost=int((before&~success).sum()),previous_failures_recovered=int((~before&success).sum()),visible_points=int((valid&visible).sum()),visible_correct=int((success&visible).sum()),occluded_points=int((valid&~visible).sum()),occluded_correct=int((success&~visible).sum()),RF_success=success.tolist(),RF_error=errors.tolist(),reference_success=before.tolist()))
    sums={k:sum(r[k] for r in result) for k in ['hands','matched_hands','points','RF_correct','SAM_correct','previous_correct_lost','previous_failures_recovered','visible_points','visible_correct','occluded_points','occluded_correct']};groups=sorted(set(r['group'] for r in result));delta=[sum(r['RF_correct']-r['SAM_correct'] for r in result if r['group']==g)/sum(r['points'] for r in result if r['group']==g) for g in groups];rng=np.random.default_rng(52)
    sums.update(images=len(result),groups=len(groups),RF_PCK10=sums['RF_correct']/sums['points'],SAM_PCK10=sums['SAM_correct']/sums['points'],paired_source_PCK_delta_CI95=np.quantile(rng.choice(delta,(5000,len(groups))).mean(1),[.025,.975]).tolist(),same_full_3D_core=True,reference='v51 strong mask/RGB localizer and parameter-DiT/FK; YOLO+SAM2 vs RF box+mask',no_3D_GT=True,no_temporal_accuracy_evidence=True,GT_used_for_inference=False,default_replaced=False,outputs_frozen_before_labels=True)
    sums['mask_only_replacement']=args.folder=='glove_yolo_rf_complete'
    (ROOT/(args.name+'_evaluation.json')).write_text(json.dumps(sums,indent=2));(ROOT/(args.name+'_per_image.json')).write_text(json.dumps(result));print(json.dumps(sums),flush=True)

if __name__=='__main__':main()
