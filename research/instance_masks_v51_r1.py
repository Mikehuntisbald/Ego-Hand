"""Prediction-only SAM2 hand masks and conservative per-instance association."""
import sys
from pathlib import Path
import cv2,numpy as np,torch
from scipy.optimize import linear_sum_assignment

THIRD=Path('/mnt/why/hot3d_hand_residual/third_party_v51')
sys.path[:0]=[str(THIRD/'sam2'),str(THIRD/'deps')]

class HandInstanceSegmenter:
    def __init__(self,device='cuda:0'):
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml','/mnt/why/HOT3D/domain_data_v51/sam2.1_hiera_small.pt',device=device,apply_postprocessing=False)
        self.predictor=SAM2ImagePredictor(model);self.device=device

    @torch.inference_mode()
    def __call__(self,image,boxes,anchors=None):
        boxes=np.asarray(boxes,np.float32).reshape(-1,4)
        if len(boxes)==0:return []
        self.predictor.set_image(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        predicted=[];qualities=[]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for i,box in enumerate(boxes):
                positive=[((box[:2]+box[2:])/2).tolist()] if anchors is None else list(anchors[i])
                # Another hand's centre is a negative prompt only when it is
                # outside this ROI, so an overlap never blindly removes a palm.
                negative=[]
                for j,b in enumerate(boxes):
                    if j==i:continue
                    centre=(b[:2]+b[2:])/2;own_centre=(box[:2]+box[2:])/2
                    outside=not (box[0]<centre[0]<box[2] and box[1]<centre[1]<box[3])
                    nested=(b[0]>=box[0] and b[1]>=box[1] and b[2]<=box[2] and b[3]<=box[3]) or (box[0]>=b[0] and box[1]>=b[1] and box[2]<=b[2] and box[3]<=b[3])
                    if outside or (nested and np.linalg.norm(centre-own_centre)>.15*max(box[2:]-box[:2])):negative.append(centre)
                points=np.asarray(positive+negative,np.float32).reshape(-1,2)
                labels=np.asarray([1]*len(positive)+[0]*len(negative),np.int32)
                masks,scores,_=self.predictor.predict(box=box,point_coords=points if len(points) else None,point_labels=labels if len(points) else None,multimask_output=True)
                k=int(np.argmax(scores));predicted.append(masks[k].astype(bool));qualities.append(float(np.clip(scores[k],0,1)))
        # Shared pixels carry uncertain ownership, rather than arbitrarily
        # assigning the overlap to the first or the highest-score detection.
        raw=np.stack(predicted);conflict=raw.sum(0)>1;exclusive=raw&~conflict[None]
        union=exclusive.any(0);result=[]
        for i,mask in enumerate(exclusive):
            area=int(raw[i].sum());retained=float(mask.sum()/max(area,1))
            quality=qualities[i]*retained if mask.sum()>=16 else 0.
            result.append(dict(mask=mask,other=union&~mask,quality=quality,sam_predicted_iou=qualities[i],conflict_fraction=1-retained,quality_calibrated=False,review_required=retained<.65 or quality<.5))
        return result

def cell_conditions(own,other,positions,image_size):
    """Sample the exact image coordinates associated with native ViT cells."""
    w,h=image_size;xy=np.asarray(positions,np.float32)*np.asarray([w,h],np.float32)
    def sample(mask):return cv2.remap(mask.astype(np.float32),xy[:,0,None],xy[:,1,None],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT).reshape(-1)
    return sample(own),sample(other)

class InstanceTracker:
    """GT-free matching, with a new segment when identity is ambiguous."""
    def __init__(self,max_gap_s=.55):self.max_gap=max_gap_s;self.next_id=0;self.active={}
    @staticmethod
    def overlap(a,b):
        intersection=(a&b).sum();return float(intersection/max((a|b).sum(),1))
    def update(self,timestamp,detections):
        active={k:v for k,v in self.active.items() if 0<timestamp-v['time']<=self.max_gap}
        keys=list(active);cost=np.full((len(keys),len(detections)),1e3,float)
        for row,key in enumerate(keys):
            old=active[key]
            for col,new in enumerate(detections):
                if old.get('person_id') is not None and new.get('person_id') is not None and old['person_id']!=new['person_id']:continue
                a=np.asarray(old['box']);b=np.asarray(new['box']);scale=max(np.linalg.norm(a[2:]-a[:2]),1)
                movement=np.asarray(old.get('velocity',[0.,0.]))*(timestamp-old['time'])
                a=a+np.r_[movement,movement]
                distance=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)/scale
                if distance>1.5:continue
                appearance=0.
                if 'appearance' in old and 'appearance' in new:
                    x=np.asarray(old['appearance']);y=np.asarray(new['appearance']);appearance=1-float(x@y/max(np.linalg.norm(x)*np.linalg.norm(y),1e-9))
                warped=cv2.warpAffine(old['mask'].astype(np.uint8),np.array([[1,0,movement[0]],[0,1,movement[1]]],np.float32),(old['mask'].shape[1],old['mask'].shape[0]),flags=cv2.INTER_NEAREST).astype(bool)
                mask_iou=self.overlap(warped,new['mask'])
                cost[row,col]=.45*(1-mask_iou)+.35*min(distance,2)+.2*appearance
        assigned={};ambiguous=set()
        if cost.size:
            rows,cols=linear_sum_assignment(cost)
            for row,col in zip(rows,cols):
                value=cost[row,col]
                alternatives=np.r_[np.delete(cost[row],col),np.delete(cost[:,col],row)]
                if value>.8:continue
                if len(alternatives) and float(alternatives.min()-value)<.08:
                    ambiguous.add(col);continue
                assigned[col]=keys[row]
        result=[]
        for col,d in enumerate(detections):
            key=assigned.get(col)
            if key is None:key=self.next_id;self.next_id+=1
            result.append(dict(track_id=key,association_uncertain=col in ambiguous,**d))
            velocity=[0.,0.]
            if key in self.active:
                previous=self.active[key];dt=timestamp-previous['time']
                if dt>0:
                    before=np.asarray(previous['box']);now=np.asarray(d['box'])
                    velocity=((now[:2]+now[2:]-before[:2]-before[2:])/(2*dt)).tolist()
            active[key]=dict(d,time=timestamp,velocity=velocity)
        self.active=active;return result
