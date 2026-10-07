"""Frozen RF-DETR predictions replace SAM2 conditions; existing full 3D core."""
import argparse, json, copy
from pathlib import Path
import cv2, numpy as np, torch
import complete_instance_v51_r5 as core
from instance_masks_v51_r1 import InstanceTracker

def main():
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--threshold',type=float,required=True);p.add_argument('--video',action='store_true');p.add_argument('--source-coarse');a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(0);device='cuda:0'
    evidence=json.loads(Path(a.predictions).read_text());assert evidence['GT_free'];folder=Path(a.output).parent;folder.mkdir(parents=True,exist_ok=True)
    frames=evidence['frames'];lookup={};tracks={};tracker=InstanceTracker();association=[]
    visual=None;captured={}
    if a.video:
        import wilor_eval_common as common
        visual,_=common.load_model(device)
        handle=visual.backbone.register_forward_hook(lambda module,inputs,output:captured.update(feature=output[-1]))
    for i,f in enumerate(frames):
        d=np.load(f['output']);keep=d['scores']>=a.threshold;masks=d['masks'][keep];boxes=d['boxes'][keep];scores=d['scores'][keep]
        conflict=masks.sum(0)>1 if len(masks) else np.zeros(cv2.imread(f['image']).shape[:2],bool);exclusive=masks&~conflict[None];union=exclusive.any(0)
        qualities=[float(s*max(m.sum(),1)/max(raw.sum(),1)) for s,m,raw in zip(scores,exclusive,masks)]
        lookup[f['image']]=dict(boxes=boxes,masks=exclusive,qualities=qualities,union=union)
        meta={k:v for k,v in f.items() if k in ['image','timestamp_s','camera','camera_calibrated','source_image','image_transform']};meta.setdefault('timestamp_s',0.)
        assert cv2.imread(f['image']).shape[:2]==(1408,1408), 'Normalize camera and image before RF inference'
        if not a.video:
            for j,(b,s) in enumerate(zip(boxes,scores)):tracks[f'{i}_{j}']=dict(id=f'{i}_{j}',frames=[dict(meta,box_xyxy=b.tolist(),box_confidence=float(s),RF_prediction_index=j)])
        else:
            detections=[];rgb=cv2.imread(f['image']);descriptors=[]
            if len(boxes):
                crops=[common.crop(rgb,b,meta['camera'],padding=1.3)[0] for b in boxes]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):visual({'img':common.input_tensor(crops,[1]*len(crops),1).to(device)})
                descriptors=captured['feature'].float().mean((2,3)).cpu().numpy()
            for j,(b,s,m) in enumerate(zip(boxes,scores,exclusive)):
                detections.append(dict(box=b.tolist(),mask=m,appearance=descriptors[j],score=float(s),RF_prediction_index=j))
            for obs in tracker.update(meta['timestamp_s'],detections):
                key=obs['track_id'];tracks.setdefault(key,dict(id=f'RF_instance_{key}',frames=[]));tracks[key]['frames'].append(dict(meta,box_xyxy=obs['box'],box_confidence=obs['score'],association_uncertain=obs['association_uncertain'],RF_prediction_index=obs['RF_prediction_index']))
                association.append(dict(image=f['image'],track_id=key,uncertain=obs['association_uncertain']))
    if a.video:handle.remove();del visual;torch.cuda.empty_cache()
    source=dict(image_size=[1408,1408],tracks=list(tracks.values()))
    if a.source_coarse:
        from scipy.optimize import linear_sum_assignment
        from benchmark_rfdetr_v52 import box_iou
        source=json.loads(Path(a.source_coarse).read_text());new_lookup={};references={}
        by_original={f.get('source_image',f['image']):f for f in frames}
        for tr in source['tracks']:
            for f in tr['frames']:references.setdefault(f['image'],[]).append(f)
        for path,items in references.items():
            f=by_original[items[0].get('source_image',path)];data=np.load(f['output']);boxes=np.array([v['box_xyxy'] for v in items]);iou=box_iou(boxes,data['boxes']);indices=linear_sum_assignment(-iou);shape=data['masks'].shape[1:];masks=np.zeros((len(boxes),*shape),bool);scores=np.zeros(len(boxes));matched=[]
            for i,j in zip(*indices):
                if iou[i,j]<.25:continue
                masks[i]=data['masks'][j];scores[i]=data['scores'][j]*iou[i,j];matched.append(dict(YOLO_index=int(i),RF_index=int(j),box_iou=float(iou[i,j]),RF_confidence=float(data['scores'][j])))
            exclusive=masks&~(masks.sum(0)>1)[None];quality=[float(s*m.sum()/max(raw.sum(),1)) for s,m,raw in zip(scores,exclusive,masks)]
            new_lookup[path]=dict(boxes=boxes,masks=exclusive,qualities=quality,union=exclusive.any(0));association.append(dict(image=path,RF_query_matches=matched,unmatched_YOLO_masks_unknown=len(boxes)-len(matched)))
        lookup=new_lookup
    (folder/'input_tracks.json').write_text(json.dumps(source));(folder/'association.json').write_text(json.dumps(association))
    # Segmenter API stays identical, but masks originate only from frozen RF outputs.
    original=core.HandInstanceSegmenter
    class CachedRF:
        def __init__(self,*args):self.index=0;self.paths=list(dict.fromkeys(f['image'] for tr in source['tracks'] for f in tr['frames']))
        def __call__(self,image,boxes):
            data=lookup[self.paths[self.index]];self.index+=1;outputs=[]
            for box in boxes:
                distance=np.abs(data['boxes']-np.asarray(box)).max(-1);j=int(distance.argmin());assert distance[j]<1e-3
                mask=data['masks'][j];q=data['qualities'][j]
                outputs.append(dict(mask=mask,other=data['union']&~mask,quality=q,review_required=q<.5,quality_calibrated=False))
            return outputs
    if not source['tracks']:
        result=dict(tracks=[],constraint_checks_passed=True,no_detected_hand=True,GT_free=True)
    else:
        try:core.HandInstanceSegmenter=CachedRF;result=core.complete(source,device,prepared=bool(a.source_coarse),mask_folder=folder/'predicted_masks')
        finally:core.HandInstanceSegmenter=original
    result.update(frontend='YOLO boxes and identical WiLoR observations; RF masks replace SAM2' if a.source_coarse else 'RFDETR trained hand box+instance mask; SAM2 removed',RF_checkpoint=evidence['checkpoint'],RF_threshold=a.threshold,RF_to_YOLO_box_match_min_iou=.25 if a.source_coarse else None,GT_used_for_inference=False,identity_certified=False,default_replaced=False,association_appearance='WiLoR visual backbone descriptor' if a.video else 'independent still frames')
    Path(a.output).write_text(json.dumps(result));(folder/'freeze.json').write_text(json.dumps(dict(complete=True,GT_free=True,output_sha256=__import__('hashlib').file_digest(open(a.output,'rb'),'sha256').hexdigest())))
    print(json.dumps(dict(tracks=len(result['tracks']),constraints=result['constraint_checks_passed'])),flush=True)

if __name__=='__main__':main()
