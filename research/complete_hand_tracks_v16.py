"""YOLO26 boxes -> WiLoR with predicted side votes -> reviewed temporal3D DiT."""
import argparse,collections,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,torch
from hand3d_v8_common import V7
from ultralytics import YOLO
from dit_v3_inference import ObservationEncoder,prepare_observation
from audit_side_consensus_v16 import votes,POLICY
from compare_detectors import iou
from infer_hand3d_v16 import infer

@torch.inference_mode()
def reconstruct(source,device='cuda:2',use_provided_side=False):
    torch.set_num_threads(4);cv2.setNumThreads(0);assert source['image_size']==[1408,1408]
    records=[]
    for ti,tr in enumerate(source['tracks']):
        for fi,f in enumerate(tr['frames']):
            r={k:f[k] for k in ['image','camera','timestamp_s']};r.update(box=f['box_xyxy'],score=float(f['box_confidence']),sequence=f'inference_track{ti}',clip=f.get('clip',0),track_id=ti,track_index=ti,frame_index=fi)
            if use_provided_side:r['side_predictions']=f['side_predictions']
            records.append(r)
    if not use_provided_side:
        paths=list(dict.fromkeys(r['image'] for r in records));detector=YOLO(str(V7.parent/'detector_compare_wilor_20261003/wilor_detector.pt'));lookup={}
        for start in range(0,len(paths),32):
            part=paths[start:start+32];out=detector.predict(part,imgsz=960,device=device.split(':')[-1],batch=32,conf=.001,max_det=100,verbose=False)
            for path,p in zip(part,out):lookup[path]=dict(boxes=p.boxes.xyxy.cpu().tolist(),scores=p.boxes.conf.cpu().tolist(),classes=p.boxes.cls.cpu().tolist())
        for r in records:r['side_predictions']=lookup[r['image']]
        del detector;torch.cuda.empty_cache()
    consensus=votes(records)
    def prep(r):return prepare_observation(cv2.imread(r['image']),r['box'],r['camera'],r['side_predictions'])
    with ThreadPoolExecutor(max_workers=8) as pool:prepared=list(pool.map(prep,records))
    for r,p in zip(records,prepared):
        vote=consensus.get((r['sequence'],r['clip'],r['track_id']));right=p['right']
        if vote and vote['eligible'] and p['side_conf']<POLICY['max_center_confidence_to_override']:right=vote['right']
        r.update(original_right=p['right'],selected_right=right,side_confidence=p['side_conf'],consensus=vote)
    encoder=ObservationEncoder(device);outputs={}
    for label in ['original','selected']:
        xyz=[]
        for start in range(0,len(records),16):
            pp=prepared[start:start+16];rr=records[start:start+16];n=len(pp);pp=pp+[pp[-1]]*(16-n);rr=rr+[rr[-1]]*(16-n)
            out=encoder.features([p['wilor_crop'] for p in pp],[p['rotation'] for p in pp],[p['focal'] for p in pp],[r[f'{label}_right'] for r in rr],[p['side_conf'] for p in pp],[p['side_iou'] for p in pp],[p['reprojection_error'] for p in pp],[p['crop_fallback'] for p in pp]);xyz.append(out['wilor'][:n].cpu())
        outputs[label]=torch.cat(xyz)
    assert all(torch.isfinite(x).all() for x in outputs.values());tracks=[dict(id=tr.get('id'),frames=[]) for tr in source['tracks']]
    for k,r in enumerate(records):
        original=source['tracks'][r['track_index']]['frames'][r['frame_index']];locks=original.get('confirmed_3d',[False]*20);values=outputs['selected'][k].tolist()
        for j,lock in enumerate(locks):
            if lock:assert np.isfinite(original['xyz_camera_m'][j]).all();values[j]=original['xyz_camera_m'][j]
        # A declaration of missing existingXYZ remains review-only at output;
        # complete WiLoR predictions are used internally to give plausible
        # hypotheses instead of denoising around an arbitrary zero placeholder.
        missing=False
        if 'xyz_camera_m' in original:
            available=np.asarray(original.get('available_3d',np.isfinite(np.asarray(original['xyz_camera_m'],float)).all(-1)),bool);missing=not available.all()
        frame={key:original[key] for key in ['image','camera','timestamp_s','box_xyxy','box_confidence']};frame.update(clip=original.get('clip',0),xyz_camera_m=values,available_3d=[True]*20,confirmed_3d=locks,original_wilor_xyz_camera_m=outputs['original'][k].tolist(),predicted_right=r['selected_right'],original_predicted_right=r['original_right'],side_vote=r['consensus'],side_corrected=r['selected_right']!=r['original_right'],declared_missing_input_review_only=missing)
        tracks[r['track_index']]['frames'].append(frame)
    del encoder;torch.cuda.empty_cache()
    return dict(image_size=[1408,1408],tracks=tracks,side_policy=POLICY)

def complete(source,device='cuda:2',use_provided_side=False):
    prepared=reconstruct(source,device,use_provided_side);result=infer(prepared,device)
    for ti,tr in enumerate(result['tracks']):
        for fi,f in enumerate(tr['frames']):
            upstream=prepared['tracks'][ti]['frames'][fi];f.update(predicted_right=upstream['predicted_right'],side_corrected=upstream['side_corrected'],side_vote=upstream['side_vote'])
            if upstream['declared_missing_input_review_only']:
                original=source['tracks'][ti]['frames'][fi];available=np.asarray(original.get('available_3d',np.isfinite(np.asarray(original['xyz_camera_m'],float)).all(-1)),bool)
                f.update(automatic_projection_applied=False,missing_3d_input_review_only=True,projection_mode='declared_missing_xyz_review_only')
                for j,p in enumerate(f['joints']):
                    p.update(xyz_camera_m=original['xyz_camera_m'][j] if available[j] else None,input_available=bool(available[j]),correction_applied=False)
                    if not p['confirmed']:p.update(review_required=True,needs_special_review=True)
    result.update(mode='yolo_boxes_wilor_consensus_temporal3d_dit_v16',side_policy=POLICY,side_corrected_frames=sum(f['side_corrected'] for t in prepared['tracks'] for f in t['frames']),missing_input_review_only_hands=sum(f['missing_3d_input_review_only'] for t in result['tracks'] for f in t['frames']))
    return prepared,result

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--prepared-output');p.add_argument('--device',default='cuda:2');p.add_argument('--use-provided-side-predictions',action='store_true');a=p.parse_args()
    src=Path(a.input).resolve();dest=Path(a.output).resolve();mid=Path(a.prepared_output).resolve() if a.prepared_output else None
    prepared,result=complete(json.loads(src.read_text()),a.device,a.use_provided_side_predictions);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2))
    if mid:mid.parent.mkdir(parents=True,exist_ok=True);mid.write_text(json.dumps(prepared,indent=2))
    print(json.dumps(dict(output=str(dest),side_corrected_frames=result['side_corrected_frames'],confirmed=result['confirmed_checked'],missing_review_only=result['missing_input_review_only_hands'])),flush=True)

if __name__=='__main__':main()
