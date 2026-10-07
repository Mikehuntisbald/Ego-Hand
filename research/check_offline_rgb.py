import json
import wilor_eval_common
import cv2,numpy as np,torch
from offline_rgb_data import RUN,save
from offline_rgb_encoder import RGBEncoder,crop_roi,crop_image,cover,input_tensor
from offline_rgb_model import RGBKeypointCompleter
from train_offline_rgb import load,batch

torch.set_num_threads(4)
rows,data=load('cuda:0');ids=torch.tensor([i for i,r in enumerate(rows) if r['role']=='train'][:4],device='cuda:0')
b,mask=batch(data,ids,torch.full((4,),6,device='cuda:0'),torch.tensor([1,2,3,4],device='cuda:0'))
assert not b['observed'][:,6:12].any()
poison=dict(data);poison['xy']=data['xy'].clone();poison['xy'][ids,6:12]=float('nan');poison['gt']=torch.full_like(data['gt'],float('nan'))
c,_=batch(poison,ids,torch.full((4,),6,device='cuda:0'),torch.tensor([1,2,3,4],device='cuda:0'))
assert all(torch.equal(b[k],c[k]) for k in b),'Hidden coordinates or GT entered model conditions'
records=json.loads((RUN/'rgb_records.json').read_text());r=records[0]
crop=crop_image(cv2.imread(r['image']),crop_roi(r['box']));rect=np.array([0,0,150,256])
changed=crop.copy();changed[:,:150]=np.random.default_rng(3).integers(0,256,(256,150,3),dtype=np.uint8)
masked=cover(crop,rect);masked_changed=cover(changed,rect);assert np.array_equal(masked,masked_changed)
encoder=RGBEncoder().cuda().eval()
with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):features=encoder(input_tensor([masked,masked_changed],'cuda:0'))
assert torch.equal(features[0],features[1])
checks={}
for kind in ['dit','regression']:
    for rgb in [False,True]:
        model=RGBKeypointCompleter(kind,rgb).cuda()
        init=torch.load(wilor_eval_common.ROOT/f'experiments/offline_keypoint_diffusion_v1/{kind}/best.pt',map_location='cuda:0',weights_only=False)
        model.load_state_dict(init['model'],strict=False)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model.loss(b,data['gt'][ids],data['valid'][ids])
        assert torch.isfinite(loss);loss.backward()
        if rgb:assert model.visual[1].weight.grad is not None and model.visual[1].weight.grad.abs().sum()>0
        model.eval();other=dict(b);other['rgb']=torch.zeros_like(b['rgb'])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):a=model.encode(b);z=model.encode(other)
        same=all(torch.equal(x,y) for x,y in zip(a,z));assert same!=rgb
        checks[f'{"rgb" if rgb else "tracks"}_{kind}']=dict(real_data_gradient=True,rgb_used=rgb,gt_isolated=True,masked_coordinates_isolated=True)
save(RUN/'rgb_checks.json',dict(masked_pixels_cannot_change_encoder_features=True,all_old_keypoints_in_affected_frames_removed=True,models=checks))
print(json.dumps(checks,indent=2))
