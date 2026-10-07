"""Semantic verification of complete track hypotheses, labels never read."""
import json
from pathlib import Path
import cv2,numpy as np,torch
from torch import nn
from cache_domain_masks_v51 import crop_array
from cache_instance_conditions_v51 import RUN
from online_parameter_model_v47 import OnlineVisual

def main():
    torch.set_num_threads(4);cv2.setNumThreads(0);device='cuda:0';source=RUN/'natural_nail_care_final_r1'
    tracks=json.loads((source/'all_tracks_before_confirmation.json').read_text());rows=[];pixels=[]
    for key,tr in tracks.items():
        ids=np.linspace(0,len(tr['frames'])-1,min(5,len(tr['frames']))).round().astype(int)
        for index in ids:
            f=tr['frames'][index];pixels.append(crop_array(cv2.imread(f['image']),f['box_xyxy']));rows.append((key,int(index)))
    ck=torch.load(RUN/'paired_protocol/core_r1/dit_joint/best.pt',map_location='cpu',weights_only=False);visual=OnlineVisual(device,False);visual.load_tail(ck['visual_tail'])
    hc=torch.load(RUN/'paired_protocol/handness_r1/model.pt',map_location='cpu',weights_only=False);head=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,64),nn.GELU(),nn.Linear(64,1)).to(device).eval();head.load_state_dict(hc['model']);pred=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for begin in range(0,len(pixels),48):
            features=visual.encode_pixels(pixels[begin:begin+48],device).float().reshape(-1,16,12,1280)[:,4:12,3:9].mean((1,2));pred.extend(head(features).squeeze(-1).sigmoid().float().cpu().tolist())
    output={}
    for (key,index),score in zip(rows,pred):output.setdefault(key,dict(id=tracks[key]['id'],frames=len(tracks[key]['frames']),scores=[]))['scores'].append(score)
    for key,r in output.items():
        r.update(mean_probability=float(np.mean(r['scores'])),accepted=bool(np.median(r['scores'])>=hc['selection']['threshold']))
    dst=RUN/'natural_nail_care_semantic_r1';dst.mkdir(exist_ok=True);(dst/'handness.json').write_text(json.dumps(dict(tracks=output,threshold=hc['selection']['threshold'],GT_used=False,not_guaranteed_hand_identity=True),indent=2))
    print(json.dumps([r for r in output.values() if r['frames']>=8],indent=2),flush=True)

if __name__=='__main__':main()
