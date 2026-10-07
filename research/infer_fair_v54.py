"""Shared inference runtime and observation-only sources for a paired comparison."""
import argparse,collections,copy,hashlib,json
from pathlib import Path
import cv2,numpy as np,torch
from prepare_fair_v54 import ROOT,OLD,INITIAL,E,save,link,sha
from instance_masks_v51_r1 import InstanceTracker
import complete_instance_v51_r5 as core
import wilor_eval_common as common

MODES=['yolo_v43','yolo_v48','rf_v48','rf_v53','rf_v54','rf_v54_side_locked','rf_v54_best','yolo_v54_control']
def box_mask(box):
    out=np.zeros((1408,1408),bool);x1,y1,x2,y2=np.clip(np.asarray(box,dtype=int),0,1408);out[y1:y2,x1:x2]=True;return out
def source(front,tag):
    torch.set_num_threads(4);cv2.setNumThreads(0);device='cuda:0'
    dest=ROOT/'sources'/front/tag;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'freeze.json').exists():return
    manifest=ROOT/'glove_replay_rgb.json' if tag=='glove' else ROOT/'replay'/f'{tag}.json'
    frames=json.loads(manifest.read_text())['frames'];lookup={}
    if front=='rf':
        path=ROOT/('glove_RF' if tag=='glove' else 'replay_RF')/'predictions.json'
        evidence=json.loads(path.read_text());assert evidence['GT_free'];pred={r['image']:r for r in evidence['frames']}
        for f in frames:
            d=np.load(pred[f['image']]['output']);raw=d['masks'];exclusive=raw&~(raw.sum(0)>1)[None]
            lookup[f['image']]=dict(boxes=d['boxes'],scores=d['scores'],masks=exclusive,quality=np.asarray([float(s*m.sum()/max(int(r.sum()),1)) for s,m,r in zip(d['scores'],exclusive,raw)]))
    else:
        from ultralytics import YOLO
        ck=E/'dit_lowconfidence_v1/detector/weights/best.pt' if front=='yolo' else OLD/'paired_protocol/detector/balanced/weights/epoch2.pt'
        model=YOLO(str(ck))
        for begin in range(0,len(frames),16):
            batch=frames[begin:begin+16];outputs=model.predict([f['image'] for f in batch],imgsz=960,conf=.05,iou=.7,max_det=100,rect=False,batch=16,device='0',verbose=False)
            for f,d in zip(batch,outputs):lookup[f['image']]=dict(boxes=d.boxes.xyxy.cpu().numpy(),scores=d.boxes.conf.cpu().numpy())
        del model;torch.cuda.empty_cache()
        if front=='yolo_control':
            from instance_masks_v51_r1 import HandInstanceSegmenter
            segmenter=HandInstanceSegmenter(device)
            for f in frames:
                d=lookup[f['image']];out=segmenter(cv2.imread(f['image']),d['boxes']);d.update(masks=np.asarray([x['mask'] for x in out],dtype=bool).reshape(-1,1408,1408),quality=np.asarray([x['quality'] for x in out]))
            del segmenter;torch.cuda.empty_cache()
    # Original prediction-only v43-v48 motion/box-IoU association.
    # An uncertain match keeps a review flag, rather than creating self-
    # reinforcing duplicate tracks. Same rule for every frontend.
    from scipy.optimize import linear_sum_assignment
    from compare_detectors import iou
    tracks={};history={};next_id=0
    for fi,f in enumerate(frames):
        d=lookup[f['image']];now=f['timestamp_s'];assigned={};uncertain=set()
        active=[k for k,items in history.items() if 0<now-items[-1]['time']<=.51] if tag!='glove' else []
        previous=[]
        for key in active:
            items=history[key];box=np.asarray(items[-1]['box'],float).copy()
            if len(items)>1:box+=np.clip((box-np.asarray(items[-2]['box']))/max(items[-1]['time']-items[-2]['time'],.01)*(now-items[-1]['time']),-120,120)
            previous.append(box)
        if len(d['boxes']) and active:
            overlap=iou(d['boxes'],previous);ii,jj=linear_sum_assignment(-overlap)
            for i,j in zip(ii,jj):
                if overlap[i,j]>.1:
                    assigned[int(i)]=active[int(j)]
                    others=np.r_[np.delete(overlap[i],j),np.delete(overlap[:,j],i)]
                    if len(others) and overlap[i,j]-others.max()<.08:uncertain.add(int(i))
        for j,(box,score) in enumerate(zip(d['boxes'],d['scores'])):
            key=assigned.get(j)
            if key is None:key=next_id;next_id+=1;history[key]=[];tracks[key]=dict(id=f'{front}_{key}',frames=[])
            history[key].append(dict(time=now,box=box.tolist()))
            tracks[key]['frames'].append(dict(f,box_xyxy=box.tolist(),box_confidence=float(score),association_uncertain=j in uncertain,prediction_index=j))
    src=dict(image_size=[1408,1408],tracks=list(tracks.values()));save(dest/'input_tracks.json',src)
    prepared=core.reconstruct_generic(src,device) if src['tracks'] else src
    save(dest/'coarse.json',prepared)
    masks={}
    if front in ['rf','yolo_control']:
        for n,(image,d) in enumerate(lookup.items()):
            path=dest/f'masks_{n:05d}.npz';np.savez_compressed(path,masks=d['masks'],quality=d['quality'],boxes=d['boxes'])
            masks[image]=str(path)
    save(dest/'masks.json',masks)
    save(dest/'freeze.json',dict(complete=True,GT_free=True,source_sha256=sha(dest/'coarse.json'),mask_hashes={Path(p).name:sha(p) for p in masks.values()},frames=len(frames),observations=sum(len(t['frames']) for t in prepared['tracks']),association='identical predicted-box motion/IoU + base WiLoR appearance',side='predicted WiLoR helper plus whole-track consensus',labels_read=False))

def plain(source,checkpoint,folder):
    import complete_hand_tracks_online_v47 as online
    import hand3d_rollout_v8 as projection
    import parameter_candidates_v43 as candidates
    from semantic_parameter_model_v36 import SemanticParameterHand
    target=folder/'dit_joint';target.mkdir(parents=True,exist_ok=True)
    if not (target/'best.pt').exists():
        ck=torch.load(checkpoint,map_location='cpu',weights_only=False);ck.setdefault('visual_tail',None);torch.save(ck,target/'best.pt')
    old_run=online.RUN;old_head=online.SemanticParameterHand;old_proj=projection.project_fisheye624;old_cp=candidates.project_fisheye624;old_solver=online.solve_profile
    def project(x,p):return core.pinhole_project(x,p) if p.shape[-1]==4 else old_proj(x,p)
    uncalibrated=all(not f.get('camera_calibrated',True) for t in source['tracks'] for f in t['frames'])
    def solver(decoder,cache,obs,rows,config=None,profile='acc_x2',progress=None):
        cfg=dict(config or {})
        if uncalibrated:cfg['rgb_weight']=.35
        return old_solver(decoder,cache,obs,rows,cfg,profile,progress)
    try:
        online.RUN=folder;online.SemanticParameterHand=SemanticParameterHand;online.solve_profile=solver;projection.project_fisheye624=project;candidates.project_fisheye624=project
        return online.complete(source,'cuda:0',prepared=True,mode='joint',profile='acc_x2',checkpoint='best.pt')
    finally:online.RUN=old_run;online.SemanticParameterHand=old_head;online.solve_profile=old_solver;projection.project_fisheye624=old_proj;candidates.project_fisheye624=old_cp

def predict(mode,tag):
    assert mode in MODES;torch.set_num_threads(4);cv2.setNumThreads(0)
    front='rf' if mode.startswith('rf_') else 'yolo_control' if mode=='yolo_v54_control' else 'yolo'
    srcdir=ROOT/'sources'/front/tag;assert json.loads((srcdir/'freeze.json').read_text())['complete']
    out=ROOT/'inference'/mode/tag;out.mkdir(parents=True,exist_ok=True)
    if (out/'freeze.json').exists():return
    src=json.loads((srcdir/'coarse.json').read_text())
    selected_development_admitted=None;checkpoint_name='best.pt'
    if not src['tracks']:result=dict(tracks=[],constraint_checks_passed=True,no_detected_hand=True)
    elif mode in ['yolo_v43','yolo_v48','rf_v48']:
        checkpoint=E/('matched_parameter_v43/dit/best.pt' if mode=='yolo_v43' else 'online_rgb_iterative_v48/protected/best.pt')
        result=plain(src,checkpoint,ROOT/'runtime_checkpoints'/mode)
    else:
        lookup={}
        for image,path in json.loads((srcdir/'masks.json').read_text()).items():
            a=np.load(path);lookup[image]={k:a[k] for k in ['boxes','masks','quality']}
        paths=list(dict.fromkeys(f['image'] for t in src['tracks'] for f in t['frames']))
        class CachedMask:
            def __init__(self,*a):self.i=0
            def __call__(self,img,boxes):
                d=lookup[paths[self.i]];self.i+=1;union=d['masks'].any(0);result=[]
                for box in boxes:
                    delta=np.abs(d['boxes']-np.asarray(box)).max(-1);j=int(delta.argmin());assert delta[j]<1e-3
                    mask=d['masks'][j];q=float(d['quality'][j]);result.append(dict(mask=mask,other=union&~mask,quality=q,review_required=q<.5))
                return result
        old_segmenter=core.HandInstanceSegmenter;old_run=core.RUN;old_head=core.InstanceParameterHand
        arm='yolo_condition_control' if mode=='yolo_v54_control' else 'rf_condition_adapt'
        try:
            core.HandInstanceSegmenter=CachedMask
            if mode!='rf_v53':
                root=ROOT/arm;link(root/'core_r1',root/'paired_protocol/core_r1');link(OLD/'paired_protocol/localizer_isolated_r2',root/'paired_protocol/localizer_isolated_r2')
                core.RUN=root;done=json.loads((root/'core_r1/dit_joint/done.json').read_text())
                checkpoint_name='best.pt' if mode=='rf_v54_best' else 'last.pt'
                selected_development_admitted=bool(done['trained_development_admitted'] and (checkpoint_name=='best.pt' or done['selected_step']==160))
            if mode=='rf_v54_side_locked':
                class LockedHand(old_head):
                    def encode(self,b):
                        joint,memory,c,heat=super().encode(b);heat['side_logits']=torch.nn.functional.one_hot(b['predicted_right'].long(),2).float()*1000
                        return joint,memory,c,heat
                core.InstanceParameterHand=LockedHand
            result=core.complete(src,'cuda:0',checkpoint=checkpoint_name,prepared=True,mask_folder=out/'predicted_masks')
        finally:core.HandInstanceSegmenter=old_segmenter;core.RUN=old_run;core.InstanceParameterHand=old_head
    assert result['constraint_checks_passed']
    result.update(comparison_mode=mode,shared_runtime=True,GT_free=True,selected_trained_development_admitted=selected_development_admitted,checkpoint_name=checkpoint_name,full_pipeline_admitted=False,default_replaced=False)
    save(out/'prediction.json',result);save(out/'freeze.json',dict(complete=True,GT_free=True,prediction_sha256=sha(out/'prediction.json'),source_sha256=sha(srcdir/'coarse.json'),mode=mode,tag=tag,selected_trained_development_admitted=selected_development_admitted))
    print(json.dumps(dict(mode=mode,tag=tag,tracks=len(result['tracks']),passed=True)),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['source','predict']);p.add_argument('--front',choices=['rf','yolo','yolo_control']);p.add_argument('--mode',choices=MODES);p.add_argument('--tag',required=True);a=p.parse_args()
    source(a.front,a.tag) if a.stage=='source' else predict(a.mode,a.tag)
