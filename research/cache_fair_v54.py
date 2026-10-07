"""RF prediction-only crop/mask conditions; keep common training 3D observations."""
import collections, json, os, time
from pathlib import Path
import cv2, numpy as np, torch
from prepare_fair_v54 import ROOT,OLD,SOURCE,save,link
from offline_rgb_encoder import crop_roi
from instance_masks_v51_r1 import cell_conditions
import spatial_rgb_common as spatial
from scipy.optimize import linear_sum_assignment

def iou(a,b):
    a=np.asarray(a).reshape(-1,4);b=np.asarray(b).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:]);inter=np.maximum(hi-lo,0).prod(-1)
    return inter/np.maximum(np.maximum(a[:,2:]-a[:,:2],0).prod(-1)[:,None]+np.maximum(b[:,2:]-b[:,:2],0).prod(-1)[None]-inter,1e-9)
def load_predictions(folder):
    evidence=json.loads((folder/'predictions.json').read_text());assert evidence['GT_free']
    return {r['image']:r for r in evidence['frames']}
def conditions(row,records):
    d=np.load(row['output']);boxes=d['boxes'];scores=d['scores'];raw=d['masks'];mask=raw&~(raw.sum(0)>1)[None];union=mask.any(0)
    annotations=np.asarray([r['box'] for r in records]);ov=iou(annotations,boxes)
    matches={}
    if ov.size:
        a,b=linear_sum_assignment(-((ov>=.25).astype(float)+ov*.001))
        matches={int(i):int(j) for i,j in zip(a,b) if ov[i,j]>=.25}
    result=[]
    for i,r in enumerate(records):
        j=matches.get(i)
        if j is None:result.append(None);continue
        own=mask[j];other=union&~own;q=float(scores[j]*own.sum()/max(int(raw[j].sum()),1))
        result.append(dict(box=boxes[j],own=own,other=other,quality=q,RF_index=j,match_iou=float(ov[i,j])))
    return result
def rays(transform,focal,rotation):
    yy,xx=np.mgrid[:16,:12];canonical=np.stack([xx*16+37.5,yy*16+5.5],-1).reshape(192,2)
    x=np.c_[(canonical-127.5)/focal,np.ones(192)].astype(np.float32)@transform.T
    x=x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
    return x@rotation.T
def main():
    torch.set_num_threads(4);cv2.setNumThreads(0)
    if (ROOT/'cache_done.json').exists():return
    data=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True)
    records=json.loads((SOURCE/'records.json').read_text());mapping=json.loads((ROOT/'native_feature_map.json').read_text())
    pred=load_predictions(ROOT/'native_RF');rf_data=dict(data)
    for key in ['positions_bank','roi','rays_world']:rf_data[key]=data[key].clone()
    original_masks=torch.load(OLD/'mask_conditions.pt',weights_only=False,mmap=True)
    rf_masks={k:v.clone() if torch.is_tensor(v) else v for k,v in original_masks.items()}
    pixels=[];fids=[];audit=[]
    for n,(image,ids) in enumerate(mapping['images'].items()):
        unique={tuple(np.round(records[fid-1]['box'],3)):[] for fid in ids}
        for fid in ids:unique[tuple(np.round(records[fid-1]['box'],3))].append(fid)
        boxes=[dict(box=list(key)) for key in unique];out=conditions(pred[image],boxes)
        for key,item in zip(unique,out):
            for fid in unique[key]:
                if item is None:
                    rf_masks['own'][fid]=0;rf_masks['other'][fid]=0;rf_masks['quality'][fid]=0
                    audit.append(dict(fid=fid,matched=False,RGB_crop='common original observation; RF instance unknown'));continue
                roi=crop_roi(item['box']);imgs,pos,transform,focal,*_=spatial.prepare(records[fid-1],roi,[[0,0,0,0]])
                own,other=cell_conditions(item['own'],item['other'],pos,[1408,1408])
                rf_data['positions_bank'][fid]=torch.from_numpy(pos);rf_data['roi'][fid]=torch.tensor(roi/1408)
                rf_data['rays_world'][fid]=torch.tensor(rays(transform,focal,data['rotation'][fid].numpy()),dtype=data['rays_world'].dtype)
                rf_masks['own'][fid]=torch.tensor(own);rf_masks['other'][fid]=torch.tensor(other);rf_masks['quality'][fid]=item['quality']
                pixels.append(imgs[0]);fids.append(fid);audit.append(dict(fid=fid,matched=True,RF_box=item['box'].tolist(),iou=item['match_iou']))
        if n%50==0:save(ROOT/'cache_progress.json',dict(stage='native_RF_conditions',images=n+1,total=len(mapping['images'])))
    rf_root=ROOT/'rf_condition_adapt';torch.save(rf_data,rf_root/'native/inputs.pt');torch.save(rf_masks,rf_root/'mask_conditions.pt')
    np.save(rf_root/'native/overlay_pixels.npy',np.stack(pixels));save(rf_root/'native/overlay_fids.json',fids)
    ctrl=ROOT/'yolo_condition_control';link(SOURCE/'inputs.pt',ctrl/'native/inputs.pt');link(OLD/'mask_conditions.pt',ctrl/'mask_conditions.pt')
    save(ROOT/'native_condition_audit.json',audit)
    # Auxiliary 2D/visibility labels remain in separate targets.pt; only frozen
    # RF boxes/masks alter model inputs. Common original WiLoR 3D seeds are fixed.
    saved=torch.load(OLD/'surgical_pose/inputs.pt',weights_only=False,mmap=True)
    surgical_map=json.loads((ROOT/'surgical_supervision_map.json').read_text());groups=collections.defaultdict(list)
    for r in surgical_map:groups[r['image']].append(r)
    pred=load_predictions(ROOT/'surgical_RF');rf_inputs={k:v.clone() for k,v in saved['inputs'].items()}
    meta=[dict(r) for r in saved['metadata']];old_pixels=np.load(OLD/'surgical_pose/pixels.npy',mmap_mode='r');rf_pixels=np.array(old_pixels)
    frames={r['image']:r for r in json.loads((ROOT/'surgical_adaptation_rgb.json').read_text())['frames']};surgical_audit=[]
    for image,rows in groups.items():
        out=conditions(pred[image],[dict(box=r['human_box']) for r in rows])
        for r,item in zip(rows,out):
            index=r['index']
            if item is None:
                meta[index]['split']='excluded_unmatched_RF';surgical_audit.append(dict(index=index,matched=False));continue
            roi=crop_roi(item['box']);imgs,pos,transform,focal,*_=spatial.prepare(dict(frames[image],clip=0),roi,[[0,0,0,0]])
            own,other=cell_conditions(item['own'],item['other'],pos,[1408,1408])
            rf_pixels[index]=imgs[0];rf_inputs['positions'][index,8]=torch.tensor(pos);rf_inputs['roi'][index,8]=torch.tensor(roi/1408)
            rf_inputs['rays'][index,8]=torch.tensor(rays(transform,focal,np.eye(3)),dtype=rf_inputs['rays'].dtype)
            rf_inputs['instance_own'][index,8]=torch.tensor(own);rf_inputs['instance_other'][index,8]=torch.tensor(other);rf_inputs['instance_quality'][index,8]=item['quality']
            surgical_audit.append(dict(index=index,matched=True,RF_box=item['box'].tolist(),iou=item['match_iou']))
    assert {r['group'] for r in meta if r['split']=='train'}.isdisjoint({r['group'] for r in meta if r['split']=='dev'})
    for arm,inp,pix in [('yolo_condition_control',saved['inputs'],old_pixels),('rf_condition_adapt',rf_inputs,rf_pixels)]:
        folder=ROOT/arm/'surgical_pose';torch.save(dict(inputs=inp,metadata=meta,coarse_input_only=True,training_common_original_WiLoR_seeds=True),folder/'inputs.pt')
        if arm=='yolo_condition_control':link(OLD/'surgical_pose/pixels.npy',folder/'pixels.npy')
        else:np.save(folder/'pixels.npy',pix)
        save(folder/'cache_done.json',dict(complete=True,GT_3D=False,matched_instances=sum(r['matched'] for r in surgical_audit),single_frame_external_supervision=True))
    save(ROOT/'surgical_condition_audit.json',surgical_audit)
    save(ROOT/'cache_done.json',dict(complete=True,native_RF_matched=len(fids),native_total=len(audit),external_matched=sum(r['matched'] for r in surgical_audit),external_total=len(surgical_audit),GT_never_inference_input=True,coarse_training_observations_fixed=True))
    print((ROOT/'cache_done.json').read_text(),flush=True)
if __name__=='__main__':main()
