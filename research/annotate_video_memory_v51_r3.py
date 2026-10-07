"""GT-free offline SAM2 memory -> stable instances -> complete 3D pipeline."""
import sys,json,hashlib
from pathlib import Path
import cv2,numpy as np,torch
from torchvision.ops import nms
from instance_masks_v51 import THIRD
from cache_instance_conditions_v51 import RUN
from complete_instance_v51_r5 import complete

def main():
    torch.set_num_threads(4);cv2.setNumThreads(0);device='cuda:0';old=RUN/'natural_nail_care_integrated';folder=RUN/'natural_nail_care_video_memory_r3';folder.mkdir(exist_ok=True)
    rows=json.loads((old/'unfiltered_detection_candidates.json').read_text());baseline=json.loads((RUN/'natural_nail_care_final_r1/all_tracks_before_confirmation.json').read_text())
    global_count={}
    for track in baseline.values():
        for f in track['frames']:global_count[round(f['timestamp_s'],6)]=global_count.get(round(f['timestamp_s'],6),0)+1
    candidates=[]
    for index,row in enumerate(rows):
        count=global_count[round(row['timestamp_s'],6)];boxes=np.asarray(row['boxes']);scores=np.asarray(row['scores']);p=np.asarray(row['handness'])
        parents=[i for i in range(count) if scores[i]>=.25 and p[i]>=.8];chosen=[];anchors=[];splits=0
        for i in parents:
            parent=boxes[i];edge=max(parent[2:]-parent[:2]);area=np.prod(parent[2:]-parent[:2]);centre=(parent[:2]+parent[2:])/2
            child=[]
            for j in range(count,len(boxes)):
                box=boxes[j];inside=(box[:2]>=parent[:2]-.01*edge).all() and (box[2:]<=parent[2:]+.01*edge).all()
                ratio=np.prod(box[2:]-box[:2])/area;distance=np.linalg.norm((box[:2]+box[2:])/2-centre)/edge
                if inside and .1<ratio<.65 and .2<distance<.6 and scores[j]>=.25 and p[j]>=.8:child.append(j)
            if child:
                j=max(child,key=lambda j:scores[j]);box=boxes[j];rectangles=[np.r_[parent[:2],[parent[2],box[1]]],np.r_[[parent[0],box[3]],parent[2:]],np.r_[parent[:2],[box[0],parent[3]]],np.r_[[box[2],parent[1]],parent[2:]]]
                remaining=max(rectangles,key=lambda b:np.maximum(b[2:]-b[:2],0).prod());centre=(remaining[:2]+remaining[2:])/2
                chosen.append(j);anchors.append(((box[:2]+box[2:])/2).tolist());splits+=1
            chosen.append(i);anchors.append(centre.tolist())
        if chosen:candidates.append((splits,len(parents),sum(scores[chosen]),index,chosen,anchors))
    assert candidates,'No confident inferred seed; needs review, not forced hand count'
    splits,parents,score,seed,chosen,anchors=max(candidates,key=lambda x:(x[0],x[1],x[2]));boxes=np.asarray(rows[seed]['boxes'])[chosen]
    # A box centre may lie on a tool. Choose a point deep inside a predicted
    # foreground region, without using a hand colour or an annotated point.
    from instance_masks_v51 import HandInstanceSegmenter
    segmenter=HandInstanceSegmenter(device);image=cv2.imread(rows[seed]['image']);segmenter.predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB));anchors=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i,box in enumerate(boxes):
            options,confidence,_=segmenter.predictor.predict(box=box,multimask_output=True)
            region=np.zeros(image.shape[:2],bool);x,y,x2,y2=np.asarray(box).astype(int);region[max(0,y):min(len(region),y2),max(0,x):min(region.shape[1],x2)]=True
            options=options.astype(bool)
            scores=[]
            for mask,q in zip(options,confidence):
                overlap=(mask&region).sum();outside=(mask&~region).sum()/max(mask.sum(),1);scores.append(overlap/max(region.sum(),1)-.5*outside+.1*float(q))
            foreground=options[int(np.argmax(scores))]&region
            for j,other in enumerate(boxes):
                if j!=i and (other[:2]>=box[:2]).all() and (other[2:]<=box[2:]).all():
                    a,b,c,d=other.astype(int);foreground[max(0,b):min(len(region),d),max(0,a):min(region.shape[1],c)]=False
            distance=cv2.distanceTransform(foreground.astype(np.uint8),cv2.DIST_L2,5);assert distance.max()>0
            yy,xx=np.unravel_index(distance.argmax(),distance.shape);anchors.append([int(xx),int(yy)])
    del segmenter;torch.cuda.empty_cache()
    from online_parameter_model_v47 import OnlineVisual
    from train_mask_localizer_v51_r2 import MaskLocalizer
    from offline_rgb_encoder import crop_roi
    from instance_masks_v51 import cell_conditions
    from annotate_instances_v51_r6 import normalize_frame
    import spatial_rgb_common as spatial
    source_frames=json.loads((RUN/'natural_nail_care/input.json').read_text())['frames'];temp=folder/'seed_frame';temp.mkdir(exist_ok=True);seed_frame=normalize_frame(source_frames[seed],temp,seed)
    visual=OnlineVisual(device,False);ck=torch.load(RUN/'paired_protocol/core_r1/dit_joint/best.pt',weights_only=False,map_location='cpu');visual.load_tail(ck['visual_tail']);localizer=MaskLocalizer().to(device).eval();localizer.load_state_dict(torch.load(RUN/'paired_protocol/localizer_isolated_r2/best.pt',weights_only=False,map_location=device)['model']);segmenter=HandInstanceSegmenter(device);positive_sets=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for i,box in enumerate(boxes):
            contains_child=any(j!=i and (b[:2]>=box[:2]).all() and (b[2:]<=box[2:]).all() for j,b in enumerate(boxes))
            if contains_child:positive_sets.append([anchors[i]]);continue
            crops,positions,*_=spatial.prepare(dict(image=seed_frame['image'],camera=seed_frame['camera'],clip=0),crop_roi(box),[[0,0,0,0]])
            raw=segmenter(image,[box])[0];own,other=cell_conditions(raw['mask'],raw['other'],positions,[1408,1408]);rgb=visual.encode_pixels(crops,device)
            predicted=localizer(rgb,torch.tensor(positions,device=device)[None],torch.tensor(crop_roi(box)/1408,device=device)[None],torch.tensor(own,device=device)[None],torch.tensor(other,device=device)[None])['xy'][0].float().cpu().numpy()*1408
            points=predicted[[8,11,14,17]];points=np.maximum(points,box[:2]);points=np.minimum(points,box[2:]-1)
            positive_sets.append(points.tolist());anchors[i]=points.mean(0).tolist()
    del visual,localizer,segmenter;torch.cuda.empty_cache()
    seed_info=dict(frame=seed,timestamp_s=rows[seed]['timestamp_s'],inferred_instances=len(chosen),nested_splits=splits,boxes=boxes.tolist(),anchors=anchors,positive_point_sets=positive_sets,GT_used=False,manual_hand_count_used=False,selection='Global confident semantic hand proposals plus best interior nested split; future frame chosen before propagation',target_video_used_for_presence_training=True)
    (folder/'seed.json').write_text(json.dumps(seed_info,indent=2));print(json.dumps(seed_info),flush=True)
    from sam2.build_sam import build_sam2_video_predictor
    predictor=build_sam2_video_predictor('configs/sam2.1/sam2.1_hiera_s.yaml','/mnt/why/HOT3D/domain_data_v51/sam2.1_hiera_small.pt',device=device,apply_postprocessing=False)
    masks={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=predictor.init_state(str(old/'normalized'),offload_video_to_cpu=True)
        for obj,(box,positive) in enumerate(zip(boxes,anchors)):
            points=np.asarray(positive_sets[obj]+[a for j,a in enumerate(anchors) if j!=obj],np.float32);labels=np.asarray([1]*len(positive_sets[obj])+[0]*(len(anchors)-1),np.int32)
            predictor.add_new_points_or_box(state,seed,obj,points=points,labels=labels,box=box)
        for reverse in [False,True]:
            for frame,ids,logit in predictor.propagate_in_video(state,start_frame_idx=seed,reverse=reverse):
                logits=logit[:,0].float().cpu().numpy();order=np.argsort(ids);logits=logits[order];raw=logits>0;conflict=raw.sum(0)>1
                # Ownership remains unknown where different logits are close.
                ranking=np.sort(logits,axis=0);winner=logits.argmax(0);clear=(ranking[-1]-ranking[-2])>=1 if len(logits)>1 else np.ones_like(conflict)
                exclusive=raw.copy()
                for i in range(len(logits)):exclusive[i]&=~conflict|(clear&(winner==i))
                quality=[]
                for i in range(len(logits)):
                    retained=exclusive[i].sum()/max(raw[i].sum(),1);prob=1/(1+np.exp(-np.clip(logits[i],-30,30)));quality.append(float(prob[raw[i]].mean()*retained) if raw[i].any() else 0.)
                masks[frame]=(raw,exclusive,quality)
                if frame%30==0:print(json.dumps(dict(stage='video_instance_memory',frame=frame,reverse=reverse)),flush=True)
    del state,predictor;torch.cuda.empty_cache();source=json.loads((RUN/'natural_nail_care/input.json').read_text());normal=json.loads((old/'input_tracks.json').read_text())['tracks'][0]['frames'];lookup={round(f['timestamp_s'],6):f for f in normal}
    # Complete camera metadata comes from the normalization-only path, not labels.
    from annotate_instances_v51_r6 import normalize_frame
    normalized=folder/'normalized';normalized.mkdir(exist_ok=True);frames=[normalize_frame(f,normalized,i) for i,f in enumerate(source['frames'])]
    records=[]
    for obj in range(len(chosen)):
        track=dict(id=f'video_instance_{obj}',frames=[]);present=[]
        for i,f in enumerate(frames):
            raw,exclusive,quality=masks[i];yy,xx=np.where(raw[obj]);visible=len(xx)>=16
            if visible:box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)];present.append((i,box))
        if not present:continue
        for i,f in enumerate(frames):
            raw,exclusive,quality=masks[i];yy,xx=np.where(raw[obj]);visible=len(xx)>=16
            box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)] if visible else min(present,key=lambda x:abs(x[0]-i))[1]
            track['frames'].append(dict(f,box_xyxy=box,box_confidence=float(quality[obj]),box_confidence_scope='Uncalibrated predicted video-mask quality, not detector confidence',association_uncertain=not visible,video_mask_quality=quality[obj],video_mask_visible=visible))
            np.savez_compressed(folder/f'mask_{i:06d}_{obj:02d}.npz',own=exclusive[obj],other=exclusive[np.arange(len(chosen))!=obj].any(0))
        records.append(track)
    request=dict(image_size=[1408,1408],tracks=records);(folder/'input_tracks.json').write_text(json.dumps(request));(folder/'association_summary.json').write_text(json.dumps(dict(source='SAM2 video memory, forward/backward from predicted seeds',tracks=len(records),GT_identity_used=False,inferred_seed_count_not_GT_count=True,all_original_candidates_retained=str(old/'unfiltered_detection_candidates.json'),person_ID_certified=False),indent=2))
    # Cache-aware segmenter supplies predicted video masks to the entire core.
    import complete_instance_v51_r5 as engine
    original=engine.HandInstanceSegmenter
    class CachedMasks:
        def __init__(self,*args):self.index=0
        def __call__(self,image,boxes):
            i=self.index;self.index+=1;raw,exclusive,quality=masks[i];result=[]
            for obj in range(len(records)):
                result.append(dict(mask=exclusive[obj],other=exclusive[np.arange(len(chosen))!=obj].any(0),quality=quality[obj],review_required=quality[obj]<.5))
            return result
    try:
        engine.HandInstanceSegmenter=CachedMasks;result=complete(request,device,mask_folder=folder/'predicted_masks')
    finally:engine.HandInstanceSegmenter=original
    result.update(instance_association='Noncausal SAM2 video-memory prediction',GT_identity_used=False,person_ID_certified=False,camera_calibrated=False,metric_depth_ambiguity=True,target_video_scene_adapted=True)
    (folder/'prediction.json').write_text(json.dumps(result));print(json.dumps(dict(complete=True,tracks=len(records),constraints=result['constraint_checks_passed'])),flush=True)

if __name__=='__main__':main()
