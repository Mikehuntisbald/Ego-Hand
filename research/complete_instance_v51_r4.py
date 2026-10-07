"""Full mask-conditioned parameter DiT and existing constrained trajectory fit.

Supports actual calibrated fisheye or pinhole cameras. Noncalibrated external
images have scale-ambiguous 3D estimates and must remain review candidates.
"""
import argparse,copy,json
from pathlib import Path
import cv2,numpy as np,torch
from instance_masks_v51 import HandInstanceSegmenter,cell_conditions
from instance_parameter_model_v51_r2 import InstanceParameterHand
from train_mask_localizer_v51_r2 import MaskLocalizer
from cache_instance_conditions_v51 import RUN
import complete_hand_tracks_online_v47 as online
import complete_hand_tracks_stable_v42 as engine
import hand3d_rollout_v8 as projection
import parameter_candidates_v43 as candidates
import wilor_eval_common as common

def pinhole_project(x,camera):
    f=camera[:,:2];center=camera[:,2:4]
    return x[:,:,:2]/x[:,:,2:3].clamp_min(1e-6)*f[:,None]+center[:,None]

def reconstruct_generic(source,device):
    """WiLoR reconstruction without the legacy fisheye-only coarse aux path."""
    from ultralytics import YOLO
    from compare_detectors import iou
    detector=YOLO('/mnt/why/HOT3D/experiments/detector_compare_wilor_20261003/wilor_detector.pt')
    paths=list(dict.fromkeys(f['image'] for tr in source['tracks'] for f in tr['frames']));sides={}
    for begin in range(0,len(paths),16):
        batch=paths[begin:begin+16];out=detector.predict(batch,imgsz=960,conf=.001,max_det=100,batch=16,device=device.split(':')[-1],verbose=False)
        for path,p in zip(batch,out):sides[path]=dict(boxes=p.boxes.xyxy.cpu().numpy(),scores=p.boxes.conf.cpu().numpy(),classes=p.boxes.cls.cpu().numpy())
    del detector;torch.cuda.empty_cache();model,_=common.load_model(device);result=copy.deepcopy(source)
    for track in result['tracks']:
        images=[];rotations=[];focals=[];rights=[]
        for frame in track['frames']:
            image=cv2.imread(frame['image']);crop,R,f,_=common.crop(image,frame['box_xyxy'],frame['camera'],padding=1.3)
            pred=sides[frame['image']];ov=iou([frame['box_xyxy']],pred['boxes'])[0]
            if len(ov) and ov.max()>=.05:
                k=int(ov.argmax());right=int(pred['classes'][k]);confidence=float(pred['scores'][k])
            else:right=1;confidence=0.
            images.append(crop);rotations.append(R);focals.append(f);rights.append(right)
            frame.update(predicted_right=right,side_prediction_confidence=confidence,side_review_required=confidence<.5)
        votes=[(1 if right else -1)*frame['side_prediction_confidence'] for right,frame in zip(rights,track['frames'])]
        consensus=int(sum(votes)>=0); strength=abs(sum(votes))/max(sum(abs(v) for v in votes),1e-9)
        rights=[consensus]*len(rights)
        for frame in track['frames']:frame.update(predicted_right=consensus,side_consensus_strength=strength,side_review_required=strength<.4)
        for begin in range(0,len(images),16):
            n=min(16,len(images)-begin)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                out=common.predict(model,images[begin:begin+n],rights[begin:begin+n],rotations[begin:begin+n],focals[begin:begin+n],rotation=1,device=device)
            for j,xyz in enumerate(out['xyz']):track['frames'][begin+j].update(xyz_camera_m=xyz.tolist(),available_3d=[True]*20,confirmed_3d=[False]*20)
    del model;torch.cuda.empty_cache();return result

def complete(source,device='cuda:0',checkpoint='best.pt',prepared=False,mask_folder=None):
    assert source['image_size']==[1408,1408]
    source=copy.deepcopy(source)
    if not prepared:source=reconstruct_generic(source,device)
    if mask_folder:
        Path(mask_folder).parent.mkdir(parents=True,exist_ok=True)
        (Path(mask_folder).parent/'coarse_observations.json').write_text(json.dumps(source))
    folder=Path(mask_folder) if mask_folder else RUN/'runtime_masks';folder.mkdir(parents=True,exist_ok=True)
    lookup={}
    for ti,tr in enumerate(source['tracks']):
        for fi,f in enumerate(tr['frames']):lookup.setdefault(f['image'],[]).append((ti,fi,f['box_xyxy']))
    segmenter=HandInstanceSegmenter(device)
    for index,(path,items) in enumerate(lookup.items()):
        outputs=segmenter(cv2.imread(path),[x[2] for x in items])
        for (ti,fi,_),out in zip(items,outputs):
            dst=folder/f'{index:06d}_{ti:04d}.npz';np.savez_compressed(dst,own=out['mask'],other=out['other'])
            source['tracks'][ti]['frames'][fi].update(instance_mask=str(dst),instance_quality=out['quality'],instance_review_required=out['review_required'])
    del segmenter;torch.cuda.empty_cache()
    old_encode=engine.encode;old_run=online.RUN;old_model=online.SemanticParameterHand;old_project=projection.project_fisheye624;old_candidate_project=candidates.project_fisheye624;old_generator=online.Generator;old_select=online.select_sequence;old_solver=online.solve_profile
    def project(x,params):return pinhole_project(x,params) if params.shape[-1]==4 else old_project(x,params)
    def encode(track,*args):
        b,chosen=old_encode(track,*args);own=[];other=[];quality=[]
        for i,frame in enumerate(track['frames']):
            with np.load(frame['instance_mask']) as mask:
                x,y=cell_conditions(mask['own'],mask['other'],b['positions'][i,8].cpu().numpy(),[1408,1408])
            own.append(x);other.append(y);quality.append(frame['instance_quality'])
        ids=torch.tensor([[i if i is not None else 0 for i in row] for row in chosen],device=b['base'].device)
        b.update(instance_own=torch.tensor(np.stack(own),device=ids.device)[ids],instance_other=torch.tensor(np.stack(other),device=ids.device)[ids],instance_quality=torch.tensor(quality,device=ids.device)[ids]*b['rgb_valid'])
        return b,chosen
    class VisibleGenerator(old_generator):
        def __init__(self,head,reference):
            super().__init__(head,reference);self.spatial=MaskLocalizer().to(device).eval()
            ck=torch.load(RUN/'paired_protocol/localizer_isolated_r2/best.pt',weights_only=False,map_location=device);self.spatial.load_state_dict(ck['model'])
        def generate(self,b,seed):
            encoded=self.head.encode(b);delta=self.head.sample_trajectory(b,encoded,10,4,seed)
            state=b['kinematic_coarse'][None]+delta;right=encoded[3]['side_logits'].argmax(-1)
            xyz=torch.stack([self.head.codec.decode(x,right)[:,8] for x in state]);mean=state.mean(0)
            self.samples.append(dict(states=state[:,:,8].float().transpose(0,1).cpu(),xyz=xyz.float().transpose(0,1).cpu(),right=right.cpu(),initial_right=b['predicted_right'].cpu(),heat=self.spatial(b['rgb_native'][:,8],b['positions'][:,8],b['roi'][:,8],b['instance_own'][:,8],b['instance_other'][:,8])['xy'].float().cpu(),visible=encoded[3]['visibility_probability'][:,8].float().cpu()))
            return dict(state=mean,xyz=self.head.codec.decode(mean,right),right=right,encoded=encoded)
    def select(samples,cache,rows):
        visibility=samples['visible'].to(cache['risk'].device)
        cache['risk']=torch.maximum(cache['risk'],(1-visibility)[:,:,None])
        return old_select(samples,cache,rows)
    uncalibrated=all(not f.get('camera_calibrated',True) for tr in source['tracks'] for f in tr['frames'])
    def solve(decoder,cache,obs,rows,config=None,profile='acc_x2',progress=None):
        cfg=dict(config or {})
        if uncalibrated:cfg['rgb_weight']=.35
        return old_solver(decoder,cache,obs,rows,cfg,profile,progress)
    try:
        online.solve_profile=solve
        online.Generator=VisibleGenerator;online.select_sequence=select
        refined=RUN/'paired_protocol/localizer_refinement_r1/core_r1/dit_joint/done.json'
        chosen_root=RUN/'paired_protocol/localizer_refinement_r1/core_r1' if refined.exists() and json.loads(refined.read_text())['trained_development_admitted'] else RUN/'paired_protocol/core_r1'
        online.RUN=chosen_root;online.SemanticParameterHand=InstanceParameterHand;engine.encode=encode
        projection.project_fisheye624=project;candidates.project_fisheye624=project
        result=online.complete(source,device,prepared=True,mode='joint',profile='acc_x2',checkpoint=checkpoint)
    finally:
        online.solve_profile=old_solver
        online.Generator=old_generator;online.select_sequence=old_select
        engine.encode=old_encode;online.RUN=old_run;online.SemanticParameterHand=old_model
        projection.project_fisheye624=old_project;candidates.project_fisheye624=old_candidate_project
    result.update(mode='experimental_full_instance_v51',instance_masks_predicted_only=True,GT_masks_inputs=False,metric_accuracy_certified=False,default_replaced=False,visibility_calibrated=False,guide='Admitted mask spatial localizer; uncalibrated camera RGB weight .35; depth remains uncertain')
    for input_track,output_track in zip(source['tracks'],result['tracks']):
        for f,g in zip(input_track['frames'],output_track['frames']):
            g.update(instance_quality=f['instance_quality'],instance_review_required=f['instance_review_required'],association_uncertain=f.get('association_uncertain',False),camera_calibrated=f.get('camera_calibrated',True))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--prepared',action='store_true');p.add_argument('--device',default='cuda:0');p.add_argument('--checkpoint',default='best.pt');a=p.parse_args()
    result=complete(json.loads(Path(a.input).read_text()),a.device,a.checkpoint,a.prepared);Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(dict(passed=result['constraint_checks_passed'],output=a.output)),flush=True)
