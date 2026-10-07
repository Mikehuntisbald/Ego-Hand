"""Runnable live-RGB/FK-trained reconstruction core, with independent stability."""
import argparse,json
from pathlib import Path
import torch,numpy as np
from hand3d_v8_common import V7
from semantic_parameter_model_v36 import SemanticParameterHand
from density_model_v13 import DensityTrajectoryHand3D
from parameter_candidates_v43 import select_sequence
from complete_hand_tracks_acceleration_v46 import profile_config,solve_profile,bounded_sample
from online_parameter_model_v47 import OnlineVisual,RUN
from temporal_window_v44 import SHORT_OFFSETS
from offline_rgb_encoder import crop_roi
import complete_hand_tracks_stable_v42 as engine
import encode_hand3d_dense_v14 as sampling
import spatial_rgb_common as s

class Generator:
    def __init__(self,head,reference):self.head=head;self.reference=reference;self.samples=[]
    def generate(self,b,seed):
        encoded=self.head.encode(b);delta=self.head.sample_trajectory(b,encoded,10,4,seed)
        state=b['kinematic_coarse'][None]+delta;right=encoded[3]['side_logits'].argmax(-1)
        xyz=torch.stack([self.head.codec.decode(x,right)[:,8] for x in state]);mean=state.mean(0)
        ref=dict(b,rgb_native=b['reference_native']);heat=self.reference.encode(ref)[3]['xy'][:,8]
        self.samples.append(dict(states=state[:,:,8].float().transpose(0,1).cpu(),xyz=xyz.float().transpose(0,1).cpu(),
            right=right.cpu(),initial_right=b['predicted_right'].cpu(),heat=heat.float().cpu()))
        return dict(state=mean,xyz=self.head.codec.decode(mean,right),right=right,encoded=encoded)
    def consume(self):
        parts=self.samples;self.samples=[];return {k:torch.cat([p[k] for p in parts]) for k in parts[0]}

def complete(source,device='cuda:3',prepared=False,mode='joint',profile='acc_x2',checkpoint='best.pt'):
    folder=RUN/f'dit_{mode}';ckpath=folder/checkpoint
    assert mode in ['joint','frozen'];assert ckpath.exists()
    original_load,original_encode,original_solve=engine.load_models,engine.encode,engine.fit_stable
    original_sample,original_offsets=sampling.sample,sampling.OFFSETS;loaded=[]
    def load(where):
        old,risk,temp,probe,projection,encoder=original_load(where);del old
        ck=torch.load(ckpath,weights_only=False,map_location=where)
        model=SemanticParameterHand(ck['kind'],where).to(where).eval();model.load_state_dict(ck['model'])
        visual=OnlineVisual(where,False);visual.load_tail(ck['visual_tail']);visual.configure()
        reference=DensityTrajectoryHand3D('dit',True).to(where).eval()
        reference.load_state_dict(torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt',weights_only=False,map_location=where)['model'])
        generator=Generator(model,reference);loaded.append((visual,generator))
        return generator,risk,temp,probe,projection,encoder
    def encode(track,encoder,probe,projection,where):
        b,chosen=original_encode(track,encoder,probe,projection,where);b['reference_native']=b['rgb_native']
        crops=[]
        for f in track['frames']:
            images,*_=s.prepare(dict(image=f['image'],camera=f['camera'],clip=f.get('clip',0)),crop_roi(f['box_xyxy']),[[0,0,0,0]])
            crops.append(images[0])
        with torch.no_grad():native=loaded[0][0].encode_pixels(crops,where)
        ids=torch.tensor([[j if j is not None else 0 for j in row] for row in chosen],device=where)
        b['rgb_native']=native[ids]*b['rgb_valid'][:,:,None,None]
        return b,chosen
    def solve(decoder,cache,mean,rows,config):
        samples=loaded[0][1].consume();cache=dict(cache,heat_xy=samples['heat'].to(device))
        obs,path=select_sequence(samples,cache,rows)
        obs['parameter_side_outlier']=samples['right']!=samples['initial_right']
        result=solve_profile(decoder,cache,{k:v.to(device) for k,v in obs.items()},rows,config,profile)
        result['candidate_selection']=path;return result
    try:
        engine.load_models=load;engine.encode=encode;engine.fit_stable=solve
        sampling.OFFSETS=SHORT_OFFSETS;sampling.sample=lambda z,d:bounded_sample(z,d,original_sample)
        result=engine.complete(source,device,prepared)
    finally:
        engine.load_models=original_load;engine.encode=original_encode;engine.fit_stable=original_solve
        sampling.sample=original_sample;sampling.OFFSETS=original_offsets
    result.update(mode='experimental_online_rgb_core_v47',checkpoint=str(ckpath),visual_mode=mode,
        context_s=1.6,config=profile_config(profile),motion_profile=profile,default_replaced=False,
        training_scope='Online last4 RGB blocks + temporal parameter generator + differentiable FK loss; detector/IK/final stability remain independent')
    for track in result['tracks']:assert track['constraints']['limits']=={k:profile_config(profile)[k] for k in track['constraints']['limits']}
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--input',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--device',default='cuda:3');ap.add_argument('--prepared',action='store_true')
    ap.add_argument('--mode',choices=['joint','frozen'],default='joint');ap.add_argument('--profile',choices=['acc_x2','strict'],default='acc_x2')
    ap.add_argument('--checkpoint',default='best.pt');a=ap.parse_args()
    result=complete(json.loads(Path(a.input).read_text()),a.device,a.prepared,a.mode,a.profile,a.checkpoint)
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(output=str(path),passed=result['constraint_checks_passed'],mode=a.mode)),flush=True)
if __name__=='__main__':main()
