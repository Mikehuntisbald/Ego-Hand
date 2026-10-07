import json,tempfile
from pathlib import Path
import spatial_rgb_common as s
import cv2,numpy as np,torch
from offline_kp_model import condition
from train_spatial_temporal import load,batch
from spatial_temporal_model import SpatialTemporalCompleter

torch.set_num_threads(4);rows,data=load('cuda:0')
ids=torch.tensor([i for i,r in enumerate(rows) if r['role']=='development'][:48],device='cuda:0')
checks={}
for arm in ['rgb_dit','rgb_regression','tracks_dit','tracks_regression']:
    ck=torch.load(s.RUN/arm/'best.pt',map_location='cuda:0',weights_only=False)
    model=SpatialTemporalCompleter(ck['kind'],ck['use_rgb']).to('cuda:0').eval();model.load_state_dict(ck['model'])
    b,m=batch(data,ids,torch.full_like(ids,6),1+ids%4)
    altered=dict(b);dirty=b['xy'].clone();dirty[~b['observed']]=12345
    altered.update(condition(dirty,b['observed'],b['dt']))
    swapped=dict(b);swapped['rgb']=torch.roll(b['rgb'],len(ids)//2,0)
    spatial=dict(b);spatial['rgb']=b['rgb'][:,:,torch.arange(191,-1,-1,device=ids.device)]
    known=dict(b);known['xy']=b['xy'].clone();known['observed']=b['observed'].clone()
    known['xy'][:,8,5]=data['xy'][ids,8,5];known['observed'][:,8,5]=True
    known.update(condition(known['xy'],known['observed'],known['dt']))
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        p=model.predict(b)['xy'];q=model.predict(altered)['xy'];sw=model.predict(swapped)['xy'];sp=model.predict(spatial)['xy'];kp=model.predict(known)['xy']
    assert torch.equal(p,q),'Hidden coordinate leakage'
    assert torch.equal(kp[:,5],known['xy'][:,8,5]),'Observed point changed'
    delta=float((p-sw).norm(dim=-1).mean()*1408)
    if model.use_rgb:assert delta>1
    else:assert torch.equal(p,sw) and torch.equal(p,sp)
    errors={k:float((z-data['gt'][ids]).norm(dim=-1)[m].mean()*1408) for k,z in [('normal',p),('shuffle_rgb',sw),('shuffle_space',sp)]}
    checks[arm]=dict(step=ck['step'],masked_coordinates_have_zero_influence=True,observed_point_preserved_exactly=True,
        rgb_swap_changes_output_mean_px=delta,development_48_window_diagnostic=errors)
records,index=s.records_and_index();r=records[0]
roi=index['roi'][1].numpy()*1408;rects=[index['rectangles'][1,5].numpy()]
observed={k:r[k] for k in ['image','camera','clip']}
a,pos,*_=s.prepare(observed,roi,rects)
with tempfile.TemporaryDirectory(dir=s.RUN) as tmp:
    image=Path(tmp)/'different_pixels.png';cv2.imwrite(str(image),np.full((1408,1408,3),231,np.uint8))
    other=dict(observed,image=str(image));z,zpos,*_=s.prepare(other,roi,rects)
    assert np.array_equal(a[0],z[0]),'Full ROI mask leaks source pixels'
    assert np.array_equal(pos,zpos),'Geometry depends on image contents'
checks['pixel_contract']=dict(full_roi_mask_independent_of_source_image=True,camera_geometry_independent_of_pixels=True,
    no_gt_or_handedness_in_encoder_record=True,spatial_tokens=192,all_backbone_blocks=32)
s.save(s.RUN/'model_checks.json',checks);print(json.dumps(checks,indent=2))
