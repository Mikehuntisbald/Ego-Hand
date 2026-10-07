"""Real labelled glove images: RGB-only inputs, separate 2D target tensors.

Surgical release does not provide physical timestamps or camera calibration.
Its renderer's 8 fps is not capture evidence. Use single-frame supervision,
and retain HOT3D for real timed 3D trajectory supervision.
"""
import json,collections,copy
from pathlib import Path
import cv2,numpy as np,torch
from annotate_instances_v51 import normalize_frame
from complete_instance_v51 import reconstruct_generic
from instance_masks_v51 import HandInstanceSegmenter,cell_conditions
from instance_parameter_model_v51 import InstanceParameterHand
from complete_hand_tracks_stable_v42 import fit_seeds,add_parameter_windows
from online_parameter_model_v47 import OnlineVisual
from cache_instance_conditions_v51 import RUN
import encode_hand3d_dense_v14 as sampling
import complete_hand_tracks_stable_v42 as engine

# Canonical HOT3D20 mapping from the author's CMU-style 21 hand landmarks.
MAPPING=[4,8,12,16,20,0,2,3,5,6,7,9,10,11,13,14,15,17,18,19]

def main():
    torch.set_num_threads(4);cv2.setNumThreads(0);device='cuda:0';folder=RUN/'surgical_pose';folder.mkdir(exist_ok=True)
    if (folder/'cache_done.json').exists():return
    raw=Path('/mnt/why/HOT3D/domain_data_v51/surgical_hands');rows=[json.loads(x) for x in (raw/'selected_records.jsonl').read_text().splitlines()]
    # Do not touch the test RGB or point labels during adaptation.
    rows=[x for x in rows if x['split']!='test'];normal=folder/'normalized';normal.mkdir(exist_ok=True);source=dict(image_size=[1408,1408],tracks=[]);targets=[];metadata=[]
    for i,row in enumerate(rows):
        frame=normalize_frame(dict(image=row['image'],timestamp_s=0.),normal,i);transform=frame['image_transform'];scale=transform['scale'];offset=np.asarray(transform['offset'])
        for hand in row['hands']:
            x,y,z,t=hand['bbox'];box=np.array([x,y,z,t])*scale+np.tile(offset,2)
            source['tracks'].append(dict(id=hand['id'],frames=[dict(frame,box_xyxy=box.tolist(),box_confidence=1.)]))
            points=np.asarray(hand['keypoints'],float).reshape(21,3)[MAPPING];xy=(points[:,:2]*scale+offset)/1408;valid=points[:,2]>0
            targets.append(dict(xy=np.where(valid[:,None],xy,0.),valid=valid,visible=points[:,2]==2,right=int(hand['category_id']=='right')))
            metadata.append(dict(id=hand['id'],group=row['group'],clip=row['clip'],split=row['split'],ROI='human training/development crop; conditional pose diagnostic',physical_timestamp_known=False,camera_calibrated=False))
    prepared=reconstruct_generic(source,device)
    # Pixel/mask/feature inference never receives the separate targets list.
    model,risk,temp,probe,projection,encoder=engine.load_models(device);del model,risk,temp
    segmenter=HandInstanceSegmenter(device);masks={}
    for path in dict.fromkeys(f['image'] for tr in prepared['tracks'] for f in tr['frames']):
        group=[tr['frames'][0] for tr in prepared['tracks'] if tr['frames'][0]['image']==path]
        out=segmenter(cv2.imread(path),[f['box_xyxy'] for f in group])
        for f,o in zip(group,out):masks[(path,tuple(f['box_xyxy']))]=o
    del segmenter;torch.cuda.empty_cache();codec=InstanceParameterHand('dit',device).codec;inputs=[];pixels=[];previous_offsets=sampling.OFFSETS;sampling.OFFSETS=[0]*17
    coarse_xyz=torch.tensor([tr['frames'][0]['xyz_camera_m'] for tr in prepared['tracks']],device=device)
    coarse_side=torch.tensor([tr['frames'][0]['predicted_right'] for tr in prepared['tracks']],device=device)
    seeds=torch.cat([fit_seeds(codec,coarse_xyz[i:i+128],coarse_side[i:i+128]) for i in range(0,len(coarse_xyz),128)])
    import spatial_rgb_common as spatial
    from offline_rgb_encoder import crop_roi
    try:
        for i,tr in enumerate(prepared['tracks']):
            b,chosen=sampling.encode(tr,encoder,probe,projection,device);f=tr['frames'][0];right=torch.tensor([f['predicted_right']],device=device);state=seeds[i:i+1]
            R=torch.eye(3,device=device)[None];T=torch.zeros(1,3,device=device);add_parameter_windows(b,chosen,state,right,R,T)
            # Uncertain domain 3D confidence is a prior, not a 2D error label.
            b['risk_camera']=torch.full((1,20),.5,device=device);b['risk_relative']=torch.full((1,20),.5,device=device)
            m=masks[(f['image'],tuple(f['box_xyxy']))];own,other=cell_conditions(m['mask'],m['other'],b['positions'][0,8].cpu().numpy(),[1408,1408])
            b['instance_own']=torch.zeros(1,17,192,device=device);b['instance_other']=torch.zeros_like(b['instance_own']);b['instance_quality']=torch.zeros(1,17,device=device)
            b['instance_own'][0,8]=torch.tensor(own,device=device);b['instance_other'][0,8]=torch.tensor(other,device=device);b['instance_quality'][0,8]=m['quality']
            images,*_=spatial.prepare(dict(image=f['image'],camera=f['camera'],clip=0),crop_roi(f['box_xyxy']),[[0,0,0,0]])
            pixels.append(images[0]);b['camera_params']=torch.tensor([f['camera']['calibration']['projection_params']],device=device)
            inputs.append({k:v.cpu() for k,v in b.items() if torch.is_tensor(v) and k not in ['rgb_native','rgb','risk_rgb']})
            if i%20==0:print(json.dumps(dict(stage='real_glove_pose_cache',done=i+1,total=len(prepared['tracks']))),flush=True)
    finally:sampling.OFFSETS=previous_offsets
    torch.save(dict(inputs={k:torch.cat([b[k] for b in inputs]) for k in inputs[0]},metadata=metadata,coarse_input_only=True),folder/'inputs.pt')
    np.save(folder/'pixels.npy',np.stack(pixels))
    torch.save(dict(xy=torch.tensor(np.stack([t['xy'] for t in targets])).float(),valid=torch.tensor(np.stack([t['valid'] for t in targets])),visible=torch.tensor(np.stack([t['visible'] for t in targets])),right=torch.tensor([t['right'] for t in targets])),folder/'targets.pt')
    (folder/'cache_done.json').write_text(json.dumps(dict(complete=True,instances=len(targets),images=len(rows),GT_only_targets=True,GT_3D=False,single_frame_external_supervision=True,no_fake_timestamps=True,ROI_diagnostic_only=True,default_changed=False),indent=2))

if __name__=='__main__':main()
