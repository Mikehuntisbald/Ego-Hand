"""Natural RGB frames -> hand detections, predicted masks/IDs -> full 3D core."""
import argparse,json,copy
from pathlib import Path
import cv2,numpy as np,torch
from instance_masks_v51 import HandInstanceSegmenter,InstanceTracker
from complete_instance_v51 import complete
from cache_instance_conditions_v51 import RUN
import wilor_eval_common as common

def normalize_frame(frame,folder,index):
    image=cv2.imread(frame['image']);assert image is not None
    h,w=image.shape[:2];scale=1408/max(h,w);nw,nh=round(w*scale),round(h*scale);x=(1408-nw)//2;y=(1408-nh)//2
    resized=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
    output=np.zeros((1408,1408,3),np.uint8);output[y:y+nh,x:x+nw]=resized
    path=folder/f'{index:06d}.jpg';cv2.imwrite(str(path),output,[cv2.IMWRITE_JPEG_QUALITY,95])
    camera=copy.deepcopy(frame.get('camera'));calibrated=camera is not None
    if camera is None:
        f=max(h,w)*scale
        camera=dict(T_world_from_camera=dict(quaternion_wxyz=[1,0,0,0],translation_xyz=[0,0,0]),calibration=dict(image_width=1408,image_height=1408,label='unmeasured_intrinsics_review_only',serial_number='unknown',projection_model_type='PinholePlane',projection_params=[f,f,w*.5*scale+x,h*.5*scale+y]))
    else:
        calib=camera['calibration'];params=calib['projection_params'];kind=calib['projection_model_type']
        if kind=='CameraModelType.FISHEYE624' and len(params)==15:params[:3]=[params[0]*scale,params[1]*scale+x,params[2]*scale+y]
        else:params[:4]=[params[0]*scale,params[1]*scale,params[2]*scale+x,params[3]*scale+y]
        calib.update(image_width=1408,image_height=1408)
    return dict(image=str(path),camera=camera,timestamp_s=float(frame['timestamp_s']),camera_calibrated=calibrated,source_image=frame['image'],image_transform=dict(scale=scale,offset=[x,y],original_size=[w,h]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--detector');a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(0)
    request=json.loads(Path(a.input).read_text());assert request['frames'];folder=Path(a.output).parent;folder.mkdir(parents=True,exist_ok=True);normalized=folder/'normalized';normalized.mkdir(exist_ok=True)
    frames=[normalize_frame(f,normalized,i) for i,f in enumerate(request['frames'])]
    assert all(frames[i]['timestamp_s']>frames[i-1]['timestamp_s'] for i in range(1,len(frames)))
    from ultralytics import YOLO
    selection=RUN/'detector/selection.json'
    detector_path=a.detector or (json.loads(selection.read_text())['selected'] if selection.exists() else '/mnt/why/HOT3D/experiments/dit_lowconfidence_v1/detector/weights/best.pt')
    detector=YOLO(str(detector_path));boxes=[]
    for begin in range(0,len(frames),16):
        batch=frames[begin:begin+16];out=detector.predict([f['image'] for f in batch],imgsz=960,conf=.05,iou=.7,max_det=20,batch=16,device=a.device.split(':')[-1],verbose=False)
        boxes.extend([dict(boxes=x.boxes.xyxy.cpu().tolist(),scores=x.boxes.conf.cpu().tolist()) for x in out])
    del detector;torch.cuda.empty_cache();segmenter=HandInstanceSegmenter(a.device);masks=[]
    for f,b in zip(frames,boxes):masks.append(segmenter(cv2.imread(f['image']),b['boxes']))
    del segmenter;torch.cuda.empty_cache();visual,_=common.load_model(a.device);captured={}
    handle=visual.backbone.register_forward_hook(lambda module,inputs,output:captured.update(feature=output[-1]))
    tracker=InstanceTracker();tracks={};association=[]
    for f,observation,instance in zip(frames,boxes,masks):
        detections=[];crops=[]
        for box in observation['boxes']:
            crop,*_=common.crop(cv2.imread(f['image']),box,f['camera'],padding=1.3);crops.append(crop)
        if crops:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):visual({'img':common.input_tensor(crops,[1]*len(crops),1).to(a.device)})
            appearance=captured['feature'].float().mean((2,3)).cpu().numpy()
        for i,(box,confidence,m) in enumerate(zip(observation['boxes'],observation['scores'],instance)):
            detections.append(dict(box=box,mask=m['mask'],appearance=appearance[i],score=confidence,mask_quality=m['quality']))
        assigned=tracker.update(f['timestamp_s'],detections)
        for d in assigned:
            key=d['track_id'];tracks.setdefault(key,dict(id=f'instance_{key}',frames=[]))
            tracks[key]['frames'].append(dict(f,box_xyxy=d['box'],box_confidence=d['score'],association_uncertain=d['association_uncertain']))
            association.append(dict(image=f['image'],timestamp_s=f['timestamp_s'],track_id=key,box=d['box'],uncertain=d['association_uncertain'],mask_quality=d['mask_quality']))
    handle.remove();del visual;torch.cuda.empty_cache()
    source=dict(image_size=[1408,1408],tracks=list(tracks.values()))
    # Save GT-free observation/association evidence before any external labels.
    (folder/'input_tracks.json').write_text(json.dumps(source,indent=2));(folder/'association.json').write_text(json.dumps(association,indent=2))
    result=complete(source,a.device,mask_folder=folder/'predicted_masks');result.update(detector=str(detector_path),camera_calibrated=all(f['camera_calibrated'] for f in frames),metric_depth_ambiguity=not all(f['camera_calibrated'] for f in frames),person_ID_certified=False)
    Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(dict(output=a.output,tracks=len(tracks),constraints_passed=result['constraint_checks_passed'],metric_depth_ambiguity=result['metric_depth_ambiguity'])),flush=True)

if __name__=='__main__':main()
