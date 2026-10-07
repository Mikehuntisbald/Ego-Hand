"""Match hand-pretrained canonical image orientation, retain physical rays.

Only predicted handedness is used. Ground-truth side never encodes RGB.
Right-hand features remain byte-identical to the accepted cache.
"""
import collections,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,torch
from hand3d_v8_common import V7,save
from audit_side_consensus_v16 import votes,POLICY
from compare_detectors import iou
from offline_rgb_encoder import crop_roi
from spatial_rgb_model import SpatialHead
import spatial_rgb_common as s
RUN=V7.parent/'canonical_rgb_v19';SOURCE=V7.parent/'context_data_v17/control'

def main():
    torch.set_num_threads(4);cv2.setNumThreads(0);RUN.mkdir(exist_ok=True);started=time.time()
    data=torch.load(SOURCE/'dense_data.pt',weights_only=False,mmap=True);dense=json.loads((V7.parent/'dense_sampling_v13/fresh_rows.json').read_text());consensus=votes(dense);records={};right=torch.ones(len(data['world']),dtype=torch.bool)
    for fid,r in enumerate(dense,1):
        b=np.asarray(r['box'],float);c=(b[:2]+b[2:])/2;size=max(float((b[2:]-b[:2]).max())*1.3,24.);roi=np.r_[c-size/2,c+size/2].astype(np.float32);p=r['side_predictions'];overlap=iou([roi],p['boxes'])[0]
        if len(overlap) and overlap.max()>=POLICY['min_side_iou']:
            j=int(overlap.argmax());side=int(p['classes'][j]);confidence=float(p['scores'][j])
        else:side=1;confidence=0.
        vote=consensus.get((r['sequence'],r['clip'],r['track_id']))
        if vote and vote['eligible'] and confidence<POLICY['max_center_confidence_to_override']:side=vote['right']
        right[fid]=bool(side);records[fid]={k:r[k] for k in ['image','camera','box','clip']}
    original,_=s.records_and_index();old=torch.load(V7/'data.pt',weights_only=False,mmap=True);prediction_rows={r['window_index']:r for r in json.loads((V7.parent/'side_consensus_v16/audit_results.json').read_text())['sets']['train_development']['rows']}
    for k,window in enumerate(data['source_window_indices']):
        fid=int(data['feature_ids'][k,8]);r=original[int(old['feature_ids'][window,8])-1];right[fid]=bool(prediction_rows[int(window)]['candidate_right']);records[fid]={key:r[key] for key in ['image','camera','box','clip']}
    ids=torch.where(~right)[0].tolist();save(RUN/'prediction_only_encoding_plan.json',dict(left_observations=len(ids),total_observations=len(right)-1,left_ids=ids,predicted_right=right.tolist(),scope='Predicted classes only; actualcamera XYZ/GT/slots unchanged; leftRGB mirrored aspretraining'))
    bank=torch.load(SOURCE/'native_bank.pt',weights_only=False,mmap=True).clone();rgb=data['rgb_bank'].clone();risk_rgb=data['risk_rgb_bank'].clone();positions=data['positions_bank'].clone();rays_world=data['rays_world'].clone()
    full,_=s.common.load_model('cuda:3');encoder=full.backbone;del full;probe=SpatialHead().to('cuda:3').eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location='cuda:3')['model']);projection=torch.load(V7.parent/'natural_reliability_v4/sealed/risk_projection.pt',weights_only=False).to('cuda:3')
    yy,xx=np.mgrid[:16,:12];canonical=np.stack([xx*16+37.5,yy*16+5.5],-1).reshape(192,2)
    def prep(fid):
        r=records[fid];roi=crop_roi(r['box']);images,pos,transform,focal,*_=s.prepare({k:r[k] for k in ['image','camera','clip']},roi,[[0,0,0,0]])
        # Flip after90degree rotation, exactly asWiLoR. A patch center37.5
        # maps tosource217.5. Merely reversing tokens would wrongly use213.5.
        ray=np.c_[-(canonical[:,0]-127.5)/focal,(canonical[:,1]-127.5)/focal,np.ones(192)]@transform.T
        point=s.common.from_json(r['camera']).eye_to_window(ray)/1408;ray/=np.maximum(np.linalg.norm(ray,axis=-1,keepdims=True),1e-8)
        assert np.isfinite(point).all();return images[0],torch.tensor(point,dtype=torch.float32),torch.tensor(ray,dtype=torch.float32)
    with ThreadPoolExecutor(max_workers=8) as pool,torch.inference_mode():
        for start in range(0,len(ids),128):
            part_ids=ids[start:start+128];prepared=list(pool.map(prep,part_ids))
            for sub in range(0,len(prepared),16):
                pp=prepared[sub:sub+16];fids=part_ids[sub:sub+16];n=len(pp);padded=pp+[pp[-1]]*(16-n);inp=s.common.input_tensor([p[0] for p in padded],[0]*16,rotation=1).to('cuda:3')
                with torch.autocast('cuda',dtype=torch.bfloat16):feature=encoder(inp[:,:,:,32:-32])[-1].flatten(2).transpose(1,2);local=probe.features(feature).flatten(2).transpose(1,2);risk=feature.float()@projection.float()
                ix=torch.tensor(fids);bank[ix]=feature[:n].half().cpu();rgb[ix]=local[:n].half().cpu();risk_rgb[ix]=risk[:n].half().cpu();positions[ix]=torch.stack([p[1] for p in pp]);camera_rays=torch.stack([p[2] for p in pp]);rays_world[ix]=torch.einsum('nsc,nkc->nsk',camera_rays,data['rotation'][ix])
            print(json.dumps(dict(encoded=min(start+128,len(ids)),total=len(ids),seconds=time.time()-started)),flush=True)
    new={**data,'rgb_bank':rgb,'risk_rgb_bank':risk_rgb,'positions_bank':positions,'rays_world':rays_world};assert torch.equal(new['xyz_camera_bank'],data['xyz_camera_bank']) and torch.equal(new['gt'],data['gt']) and torch.equal(new['feature_ids'],data['feature_ids']);assert torch.equal(rgb[right],data['rgb_bank'][right]) and torch.equal(positions[right],data['positions_bank'][right])
    torch.save(bank,RUN/'native_bank.pt');torch.save(new,RUN/'dense_data.pt')
    for name in ['trajectory_labels.pt']:
        link=RUN/name
        if not link.exists():link.symlink_to(SOURCE/name)
    save(RUN/'ready.json',dict(complete=True,left_observations=len(ids),right_observations=int(right.sum())-1,right_features_and_geometry_exact=True,xyz_gt_slots_exact=True,physical_geometry='Leftcamera rayx sign reversed beforecrop rotationtransform; actualgridpaddingcenter retained; no naive tokenflip',scope='Training/development only; original RGB and predictedhandedness only; no addedocclusion. Risks/3Dheads must berefitted; not yet accuracy evidence.',seconds=time.time()-started));print((RUN/'ready.json').read_text(),flush=True)
if __name__=='__main__':main()
