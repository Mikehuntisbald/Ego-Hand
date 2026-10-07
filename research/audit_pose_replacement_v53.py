"""Paired old-correct-point damage, actual projections and hand-side changes."""
import json
from pathlib import Path
import numpy as np,cv2
from benchmark_rfdetr_v52 import box_iou,assignments
from build_surgical_pose_cache_v51 import MAPPING
B=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007')
BASE=Path('/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007/complete_glove_test')
EDGES=[(6,7),(7,0),(5,8),(8,9),(9,10),(10,1),(5,11),(11,12),(12,13),(13,2),(5,14),(14,15),(15,16),(16,3),(5,17),(17,18),(18,19),(19,4)]
def collect(source,prediction):
    result={};by_id={tr['id']:tr for tr in prediction['tracks']}
    for tr in source['tracks']:
        for i,f in enumerate(tr['frames']):
            p=by_id[tr['id']]['frames'][i];result.setdefault(f.get('source_image',f['image']),[]).append((f,p))
    return result
def project(item,frame):
    f,p=item;xyz=np.asarray(p['candidate_xyz_camera_m']);fx,fy,cx,cy=f['camera']['calibration']['projection_params'][:4];uv=xyz[:,:2]/xyz[:,2:]*[fx,fy]+[cx,cy]
    if f['image_transform']!=frame['image_transform']:uv=(uv-np.asarray(f['image_transform']['offset']))/f['image_transform']['scale']*frame['image_transform']['scale']+np.asarray(frame['image_transform']['offset'])
    return uv
def main():
    assert (B/'glove_complete/freeze.json').exists() and (BASE/'freeze.json').exists()
    old=collect(json.loads((BASE/'candidate/coarse_observations.json').read_text()),json.loads((BASE/'candidate_sealed.json').read_text()));new=collect(json.loads((B/'glove_complete/input_tracks.json').read_text()),json.loads((B/'glove_complete/prediction.json').read_text()));frames=json.loads((B/'glove_rgb.json').read_text())['frames'];labels={r['id']:r for r in map(json.loads,Path('/mnt/why/HOT3D/domain_data_v51/surgical_hands/selected_records.jsonl').read_text().splitlines())};scores={r['id']:r for r in json.loads((B/'full_glove_per_image.json').read_text())};details=[];gallery=[];out=B/'review';out.mkdir(exist_ok=True)
    for frame in frames:
        row=labels[frame['id']];scale=frame['image_transform']['scale'];offset=np.asarray(frame['image_transform']['offset']);gt=np.array([h['bbox'] for h in row['hands']])*scale+np.tile(offset,2);xy=np.array([np.array(h['keypoints']).reshape(21,3)[MAPPING] for h in row['hands']]);valid=xy[:,:,2]>0;xy=xy[:,:,:2]*scale+offset;lookup={}
        for name,table in [('old',old),('new',new)]:
            items=table[frame['source_image']];boxes=[]
            for f,p in items:
                box=np.asarray(f['box_xyxy']);box=(box-np.tile(f['image_transform']['offset'],2))/f['image_transform']['scale']*scale+np.tile(offset,2);boxes.append(box)
            mat=box_iou(gt,boxes);a,b=assignments(mat);lookup[name]={int(i):items[j] for i,j in zip(a,b) if mat[i,j]>=.5}
        s=scores[frame['id']];before=np.array(s['reference_success'],bool);after=np.array(s['RF_success'],bool)
        for i in range(len(gt)):
            old_item=lookup['old'].get(i);new_item=lookup['new'].get(i);lost=int((before[i]&~after[i]).sum());recovered=int((~before[i]&after[i]).sum());side_changed=bool(old_item[1]['predicted_right']!=new_item[1]['predicted_right']) if old_item and new_item else None
            details.append(dict(id=frame['id'],group=frame['group'],hand=i,old_detected=old_item is not None,new_detected=new_item is not None,old_correct=int(before[i].sum()),lost=lost,recovered=recovered,hand_side_changed=side_changed))
        images=[]
        for name,color in [('old',(255,170,0)),('new',(220,0,220))]:
            im=cv2.imread(frame['image'])
            for i in range(len(gt)):
                for point in xy[i][valid[i]]:cv2.circle(im,tuple(np.rint(point).astype(int)),5,(0,200,0),-1)
                item=lookup[name].get(i)
                if item:
                    uv=project(item,frame)
                    for a,b in EDGES:
                        if np.isfinite(uv[[a,b]]).all() and (uv[[a,b]]>=0).all() and (uv[[a,b]]<1408).all():cv2.line(im,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),color,3)
                    for p in uv:
                        if np.isfinite(p).all() and (p>=0).all() and (p<1408).all():cv2.circle(im,tuple(np.rint(p).astype(int)),4,color,-1)
            cv2.putText(im,'YOLO + SAM2 | green GT' if name=='old' else 'RF boxes + masks | same 3D core',(20,42),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);images.append(cv2.resize(im,(704,704)))
        path=out/('pose_'+frame['id']+'.jpg');cv2.imwrite(str(path),np.concatenate(images,axis=1));gallery.append(dict(id=frame['id'],image=path.name,lost=sum(v['lost'] for v in details if v['id']==frame['id']),recovered=sum(v['recovered'] for v in details if v['id']==frame['id'])))
    total_lost=sum(v['lost'] for v in details);drop_lost=sum(v['lost'] for v in details if v['old_detected'] and not v['new_detected']);side_lost=sum(v['lost'] for v in details if v['hand_side_changed']);outcome=dict(hands=len(details),correct_points_lost=total_lost,lost_from_new_detector_drop=drop_lost,lost_with_both_detectors_matched=total_lost-drop_lost,lost_on_hand_side_changed_hands=side_lost,hand_side_changed_hands=sum(v['hand_side_changed'] is True for v in details),details=details,gallery=gallery,causal_mechanism_not_proven=True,GT_free_inference=True,diagnostic_replay=True)
    (out/'pose_damage_audit.json').write_text(json.dumps(outcome,indent=2));print(json.dumps({k:v for k,v in outcome.items() if k not in ['details','gallery']}),flush=True)
if __name__=='__main__':main()
