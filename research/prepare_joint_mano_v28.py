"""Freeze v28 development inputs and cache WiLoR/MANO and RGB observations.

No previously evaluated fresh batches or retained hardcases enter this stage.
GT is written separately for evaluation, never used in track/input selection.
"""
import argparse,collections,hashlib,json,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,torch
from hand3d_v8_common import V7,save
from hand3d_data_v7 import batch,risk_features
from hand3d_risk_v7 import Risk3D
from density_model_v13 import DensityTrajectoryHand3D
from adaptive_projection_v14 import apply
from compare_detectors import iou
from dit_v3_inference import prepare_observation
from hand3d_rollout_v8 import project_fisheye624
import spatial_rgb_common as s

ROOT=V7.parent;DATA=ROOT/'side_data_v16/consensus';MODEL=ROOT/'side_native_v16';OUT=ROOT/'joint_mano_v28'

def records_bank(data):
    primary=json.loads((ROOT/'dense_sampling_v13/fresh_rows.json').read_text())
    old_records,_=s.records_and_index();old=torch.load(V7/'data.pt',weights_only=False,mmap=True)
    # The native bank appends sorted unique observation IDs, not window order.
    # Reconstruct the explicit old-window -> new-center-fid mapping.
    full=primary+[None]*(len(data['world'])-1-len(primary))
    for wi,w in enumerate(data['source_window_indices']):
        fid=int(data['feature_ids'][wi,8]);record=old_records[int(old['feature_ids'][int(w),8])-1]
        if full[fid-1] is not None:assert full[fid-1]['image']==record['image']
        full[fid-1]=record
        assert record['image']==data['rows'][wi]['image'] and record['subject']==data['subjects'][wi]
    assert len(full)==len(data['world'])-1 and all(x is not None for x in full)
    return primary,full

def assemble():
    OUT.mkdir(exist_ok=True);data=torch.load(DATA/'dense_data.pt',weights_only=False,mmap=True)
    primary,full=records_bank(data);roles=json.loads((ROOT/'dense_sampling_v13/source_roles.json').read_text())
    roles={(r['sequence'],r['clip']):r['role'] for r in roles}
    needed=[fid for fid,r in enumerate(primary,1) if roles[(r['sequence'],r['clip'])] in ['dev_select','dev_calibrate']]
    images=collections.defaultdict(list)
    for fid in needed:images[full[fid-1]['image']].append(fid)
    aliases={};center_rows={}
    for wi,role in enumerate(data['roles']):
        if role not in ['dev_select','dev_calibrate']:continue
        fid=int(data['feature_ids'][wi,8]);r=full[fid-1];candidates=images[r['image']]
        overlap=iou([r['box']],[full[x-1]['box'] for x in candidates])[0];j=int(overlap.argmax())
        assert overlap[j]>.99,'Do not guess a center track'
        target=candidates[j];assert target not in aliases
        aliases[target]=fid;center_rows[target]=wi
    groups=collections.defaultdict(list)
    for fid in needed:
        r=full[fid-1];groups[(r['sequence'],r['clip'],r['track_id'])].append((int(r['timestamp_ns']),fid))
    offsets=np.array([-120,-80,-50,-30,-10,-3,-2,-1,0,1,2,3,10,30,50,80,120])/30.
    fids=[];times=[];center_ids=[];metadata=[];labels=[];valid=[]
    for key,values in sorted(groups.items()):
        values.sort();stamps=np.array([v[0] for v in values],np.int64);ids=np.array([v[1] for v in values])
        for stamp,fid in values:
            r=full[fid-1];selected=aliases.get(fid,fid);window=center_rows.get(fid)
            if window is not None:slot=data['feature_ids'][window].numpy();dt=data['dt'][window].numpy()
            else:
                targets=stamp+np.rint(offsets*1e9).astype(np.int64);distance=np.abs(targets[:,None]-stamps[None]);nearest=distance.argmin(1)
                slot=np.where(distance[np.arange(17),nearest]<=20_000_000,ids[nearest],0);slot[8]=selected
                dt=np.where(slot>0,(stamps[nearest]-stamp)/1e9,offsets);dt[8]=0.
            fids.append(slot);times.append(dt);center_ids.append(selected)
            metadata.append(dict(sequence=r['sequence'],subject=r['subject'],clip=r['clip'],track_id=r['track_id'],
                frame=r['frame'],timestamp_ns=r['timestamp_ns'],image=r['image'],role=roles[(r['sequence'],r['clip'])],
                primary_fid=fid,center_fid=selected,window_index=window,box=full[selected-1]['box'],score=float(data['scores'][selected])))
            if window is not None:labels.append(data['gt'][window]);valid.append(data['valid'][window])
            elif r['matched']:g=torch.tensor(r['gt']);labels.append(torch.nan_to_num(g));valid.append(torch.isfinite(g).all(-1))
            else:labels.append(torch.zeros(20,3));valid.append(torch.zeros(20,dtype=torch.bool))
    fids=torch.tensor(np.array(fids));times=torch.tensor(np.array(times),dtype=torch.float32)
    params=torch.load(ROOT/'aligned_density_v13/camera_params.pt',weights_only=False,mmap=True)
    xy=project_fisheye624(data['xyz_camera_bank'],params)/1408
    observed=torch.isfinite(xy).all(-1)&(xy>=0).all(-1)&(xy<1).all(-1)&data['available_bank']
    inputs=dict(feature_ids=fids,dt=times,xy=torch.nan_to_num(xy)[fids],observed_2d=observed[fids],center_ids=torch.tensor(center_ids))
    for i,r in enumerate(metadata):
        if r['window_index'] is not None:
            wi=r['window_index'];assert torch.equal(fids[i],data['feature_ids'][wi])
            assert torch.equal(times[i],data['dt'][wi]);inputs['xy'][i]=data['xy'][wi];inputs['observed_2d'][i]=data['observed_2d'][wi]
    torch.save(inputs,OUT/'inputs.pt');torch.save(dict(gt=torch.stack(labels),valid=torch.stack(valid)),OUT/'evaluation_labels.pt')
    save(OUT/'rows.json',metadata)
    protocol=dict(stage='Frozen-v16 full-trajectory proposals + shared-shape jointMANO MAP optimization; next parameterDiT conditional on firststage evidence',
        selection='Only dev_select clips/labels select weights. Lock config before dev_calibrate evaluation. Fifth/previousfresh/retained failures excluded.',
        rows=len(metadata),centers=len(aliases),roles=dict(collections.Counter(r['role'] for r in metadata)),
        inputs='Original center slots/RGB/XYZ unchanged. Other dense centers use predictedtrack/time only. Alias center box IoU>0.99; GT never picks a hand.',
        inference_gt='GT targets deliberately poisoned before batch construction; separate evaluation labels.',
        comparisons=['v16 final currentframe policy','Sharedshape MANO fitting without temporal terms','Sharedshape MANO plus wholeclip motion/RGB/protection'],
        acceptance={'vs_original_relative_improvement_min':.05,'vs_original_relative_paired_ci_upper_max':0.,'vs_v16_camera_regression_max_mm':.1,
            'vs_v16_relative_regression_max_mm':.1,'camera_and_relative_good_harm_max':.01,'hard_recovery_no_loss':True,
            'spurious_jump_pair_reduction_min':.3,'bone_flicker_reduction_min':.3,'allframe_metrics_include_fallbacks':True},
        current_default='v16 unchanged',source_hashes={'model':hashlib.sha256((MODEL/'consensus/uniform_adaptive/best.pt').read_bytes()).hexdigest()},
        natural_only=True,artificial_occlusion=False,created_unix=time.time())
    save(OUT/'protocol.json',protocol);return data,full,inputs,metadata

@torch.inference_mode()
def cache_mano(data,full,inputs,rows,device):
    if (OUT/'mano_observations.pt').exists():return
    model,_=s.common.load_model(device);model.eval();choices={int(r['fid']):r for r in json.loads((ROOT/'side_data_v16/prediction_only_changes.json').read_text())}
    started=time.time();outputs=collections.defaultdict(list);errors=[];Q=np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32)
    for start in range(0,len(rows),16):
        part=rows[start:start+16];records=[full[r['center_fid']-1] for r in part]
        def prep(r):return prepare_observation(cv2.imread(r['image']),r['box'],r['camera'],r['side_predictions'])
        with ThreadPoolExecutor(max_workers=8) as pool:prepared=list(pool.map(prep,records))
        rights=[choices[r['center_fid']]['candidate_right'] if r['center_fid'] in choices else p['right'] for r,p in zip(part,prepared)]
        n=len(part);padded=prepared+[prepared[-1]]*(16-n);rs=rights+[rights[-1]]*(16-n)
        image=s.common.input_tensor([p['wilor_crop'] for p in padded],rs,1).to(device)
        # The audited XYZ worker ran WiLoR in FP32; only native RGB features
        # used BF16. Match that contract before caching MANO parameters.
        out=model({'img':image})
        f=torch.tensor([p['focal'] for p in padded],device=device);cam=out['pred_cam'].float()
        shift=torch.stack([cam[:,1],cam[:,2],2*f/(256*cam[:,0].clamp_min(1e-6))],-1)
        transform=np.array([p['rotation'] for p in padded])@Q.T;transform[:,:,0]*=(2*np.array(rs)-1)[:,None]
        transform=torch.tensor(transform,device=device)
        xyz=torch.einsum('bij,bkj->bki',transform,out['pred_keypoints_3d'][:,s.common.MAPPING].float()+shift[:,None])[:n]
        fid=inputs['center_ids'][start:start+n];errors.append((xyz.cpu()-data['xyz_camera_bank'][fid]).norm(dim=-1)*1000)
        outputs['xyz'].append(xyz.cpu());outputs['transform'].append(transform[:n].cpu());outputs['right'].append(torch.tensor(rights))
        for k in ['global_orient','hand_pose','betas']:outputs[k].append(out['pred_mano_params'][k][:n].float().cpu())
        if start%512==0:print(json.dumps(dict(stage='MANOinit',done=start+n,total=len(rows),seconds=time.time()-started)),flush=True)
    result={k:torch.cat(v) for k,v in outputs.items()};error=torch.cat(errors)
    stats=dict(mean_mm=float(error.mean()),p99_mm=float(error.quantile(.99)),max_mm=float(error.max()),passed=bool(error.max()<.5))
    save(OUT/'mano_reconstruction_parity.json',stats);assert stats['passed'],stats
    torch.save(result,OUT/'mano_observations.pt');del model;torch.cuda.empty_cache();print(json.dumps(dict(MANOcached=True,**stats)),flush=True)

@torch.inference_mode()
def cache_candidates(data,inputs,rows,device):
    if (OUT/'candidates.pt').exists():return
    frozen=json.loads((MODEL/'fifth_seal.json').read_text())
    for name,h in frozen['code_sha256'].items():assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==h,name
    raw=dict(data);raw.update({k:v for k,v in inputs.items() if k not in ['center_ids']})
    # Neither current nor historical GT is an observation.
    raw.update(gt=torch.full((len(rows),20,3),float('nan')),valid=torch.zeros(len(rows),20,dtype=torch.bool))
    data={k:v.to(device) if torch.is_tensor(v) else v for k,v in raw.items()};ids=torch.arange(len(rows),device=device)
    ck=torch.load(DATA/'risk_dense/risk_all.pt',weights_only=False,map_location=device);risk=Risk3D(ck['dim']).to(device).eval();risk.load_state_dict(ck['model'])
    temp=torch.tensor(json.loads((DATA/'risk_dense/calibration.json').read_text())['temperature'],device=device)
    prob=torch.cat([(risk(risk_features(batch(data,ids[start:start+64])))/temp).sigmoid() for start in range(0,len(ids),64)])
    model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(torch.load(MODEL/'consensus/uniform_adaptive/best.pt',weights_only=False,map_location=device)['model'])
    bank=torch.load(DATA/'native_bank.pt',weights_only=False,mmap=True).to(device);result=collections.defaultdict(list);started=time.time()
    for start in range(0,len(ids),8):
        ix=ids[start:start+8];b=batch(data,ix,prob);b['rgb_native']=bank[data['feature_ids'][ix]]
        with torch.autocast('cuda',dtype=torch.bfloat16):
            encoded=model.encode(b);delta=model.sample_trajectory(b,encoded,10,4,202610114+start)
            from hand3d_temporal_v7 import pack,unpack
            trajectory=unpack(pack(b['xyz'])[None]+delta).mean(0)
        proposal=trajectory[:,8].float();result['proposal'].append(proposal.cpu());result['trajectory'].append(trajectory.float().cpu())
        result['heat_logits'].append(encoded[3]['logits'][:,8].float().cpu());result['heat_xy'].append(encoded[3]['xy'][:,8].float().cpu())
        result['baseline'].append(apply(proposal,b,frozen['policy']).cpu())
        if start%512==0:print(json.dumps(dict(stage='DiT_RGB',done=start+len(ix),total=len(ids),seconds=time.time()-started)),flush=True)
    result={k:torch.cat(v) for k,v in result.items()};result['risk']=prob.cpu()
    # Existing dev_calibrate point predictions are the matched v16 control.
    existing=torch.load(MODEL/'consensus/uniform_adaptive/calibration.pt',weights_only=False,map_location='cpu')
    index={int(w):i for i,w in enumerate(existing['indices'].tolist())}
    for n,r in enumerate(rows):
        wi=r['window_index']
        if wi in index:result['proposal'][n]=existing['proposal'][index[wi]]
    b=batch(data,ids,prob);result['baseline']=apply(result['proposal'].to(device),b,frozen['policy']).cpu()
    result.update(base=data['xyz_camera_bank'][inputs['center_ids'].to(device)].cpu(),rotation=data['rotation'][inputs['center_ids'].to(device)].cpu(),
        translation=data['translation'][inputs['center_ids'].to(device)].cpu(),roi=data['roi'][inputs['center_ids'].to(device)].cpu(),
        positions=data['positions_bank'][inputs['center_ids'].to(device)].cpu(),
        camera_params=torch.load(ROOT/'aligned_density_v13/camera_params.pt',weights_only=False,mmap=True)[inputs['center_ids']],policy=frozen['policy'])
    torch.save(result,OUT/'candidates.pt');save(OUT/'ready.json',dict(complete=True,observations=len(rows),original_centers=sum(r['window_index'] is not None for r in rows),
        gt_poisoned_inference=True,visual_evidence='Frozenv16 spatialheat_logits/xy fromRGB; notvisibility truth or calibratederror',
        trajectory='All17sampledXYZ slots retained; no perpoint clip after jointMANO',default_changed=False))
    print((OUT/'ready.json').read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4);cv2.setNumThreads(0)
    d,r,i,m=assemble();cache_mano(d,r,i,m,a.device);cache_candidates(d,i,m,a.device)
