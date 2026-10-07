"""Paired full parameter sampling/selection/FK fitting; singleton poses only."""
import json,hashlib,collections
from pathlib import Path
import torch,numpy as np
from online_parameter_model_v47 import RUN as SOURCE
from parameter_codec_v31 import observation_batch,OUT
from instance_parameter_model_v51_r2 import InstanceParameterHand
from train_mask_localizer_v51_r2 import MaskLocalizer
from cache_instance_conditions_v51 import RUN
from parameter_candidates_v43 import select_sequence
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from complete_hand_tracks_acceleration_v46 import solve_profile
import hand3d_rollout_v8 as projection
import parameter_candidates_v43 as candidates
from complete_instance_v51_r1 import pinhole_project
from hand3d_v8_common import metrics

def main():
    torch.set_num_threads(4);device='cuda:0';folder=RUN/'paired_protocol/full_spatial_solver';folder.mkdir(exist_ok=True)
    if (folder/'evaluation.json').exists():return
    ck=torch.load(RUN/'paired_protocol/core_r1/dit_joint/best.pt',weights_only=False,map_location='cpu')
    model=InstanceParameterHand('dit',device).to(device).eval();model.load_state_dict(ck['model'])
    hc=torch.load(RUN/'paired_protocol/localizer_isolated_r2/best.pt',weights_only=False,map_location='cpu');head=MaskLocalizer().to(device).eval();head.load_state_dict(hc['model'])
    feature=torch.load(RUN/'paired_protocol/localizer_isolated_r2/features.pt',weights_only=False,mmap=True)
    sin=torch.load(RUN/'surgical_pose/inputs.pt',weights_only=False,mmap=True);sdev=[i for i,r in enumerate(sin['metadata']) if r['split']=='dev']
    native=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True);native={k:v.to(device) if torch.is_tensor(v) else v for k,v in native.items()}
    mask=torch.load(RUN/'mask_conditions.pt',weights_only=False);nids=mask['indices']['dev_select'];nloc=[feature['native_indices'].index(i) for i in nids]
    coarse=torch.load(SOURCE.parent/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True).to(device)
    right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False).to(device);risk=torch.load(SOURCE/'risk.pt',weights_only=False)['joint'].to(device)
    camera=torch.load(SOURCE.parent/'aligned_density_v13/camera_params.pt',weights_only=False,mmap=True).to(device)
    allout={};sealed=[]
    old=projection.project_fisheye624;oldcandidate=candidates.project_fisheye624
    projection.project_fisheye624=lambda x,c:pinhole_project(x,c) if c.shape[-1]==4 else old(x,c)
    candidates.project_fisheye624=projection.project_fisheye624
    try:
        for domain in ['surgery','native']:
            ids=sdev if domain=='surgery' else nids;pieces=collections.defaultdict(list);cacheparts=collections.defaultdict(list)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                for begin in range(0,len(ids),8):
                    ix=ids[begin:begin+8]
                    if domain=='surgery':
                        b={k:v[ix].to(device) for k,v in sin['inputs'].items()};rgb=feature['surgical'][ix].to(device)
                        b['rgb_native']=torch.zeros(len(ix),17,192,1280,device=device);b['rgb_native'][:,8]=rgb
                        R=torch.eye(3,device=device)[None].expand(len(ix),-1,-1);T=torch.zeros(len(ix),3,device=device);cam=b['camera_params']
                    else:
                        ii=torch.tensor(ix,device=device);b=observation_batch(native,ii,risk,coarse,right,preserve_fitted=True);f=native['feature_ids'][ii];center=f[:,8]
                        # Cached head evaluation only contains centre crops. Encode other slots online to retain the full trained context.
                        from online_parameter_model_v47 import OnlineVisual
                        if begin==0:
                            visual=OnlineVisual(device,False);visual.load_tail(ck['visual_tail']);pixels=np.load(SOURCE/'pixels.npy',mmap_mode='r')
                        b=visual.replace(b,f,pixels);b.update(instance_own=mask['own'][f.cpu()].to(device),instance_other=mask['other'][f.cpu()].to(device),instance_quality=mask['quality'][f.cpu()].to(device))
                        rgb=b['rgb_native'][:,8];R=native['rotation'][center];T=native['translation'][center];cam=camera[center]
                    enc=model.encode(b);delta=model.sample_trajectory(b,enc,10,4,2026100757+begin)
                    state=b['kinematic_coarse'][None]+delta;side=enc[3]['side_logits'].argmax(-1);xyz=torch.stack([model.codec.decode(x,side)[:,8] for x in state])
                    guidance=head(rgb,b['positions'][:,8],b['roi'][:,8],b['instance_own'][:,8],b['instance_other'][:,8])['xy']
                    for k,v in [('states',state[:,:,8].float().transpose(0,1)),('xyz',xyz.float().transpose(0,1)),('right',side),('initial_right',b['predicted_right']),('old_heat',enc[3]['xy'][:,8]),('new_heat',guidance)]:pieces[k].append(v.float().cpu() if k.endswith('heat') else v.cpu())
                    for k,v in [('base',b['base']),('rotation',R),('translation',T),('risk',torch.stack([b['risk_camera'],b['risk_relative']],-1)),('roi',b['roi'][:,8]),('camera_params',cam),('confirmed',b['confirmed'])]:cacheparts[k].append(v.cpu())
                if domain=='native':del visual;torch.cuda.empty_cache()
            samples={k:torch.cat(v) for k,v in pieces.items()};cache={k:torch.cat(v).to(device) for k,v in cacheparts.items()}
            rows=[dict(sequence=f'{domain}_{i}',clip=0,track_id=i,timestamp_ns=0,frame=0) for i in range(len(ids))]
            outputs={}
            for name,heat in [('reference',samples['old_heat']),('candidate',samples['new_heat'])]:
                cache['heat_xy']=heat.to(device);obs,selection=select_sequence(samples,cache,rows)
                obs['parameter_side_outlier']=samples['right']!=samples['initial_right']
                result=solve_profile(ParameterTrajectoryDecoder(device),cache,{k:v.to(device) for k,v in obs.items()},rows,profile='acc_x2')
                assert result['check']['passed'];outputs[name]=dict(xyz=result['prediction'].cpu(),check=result['check'])
            path=folder/(domain+'_sealed.pt');torch.save(dict(outputs=outputs,indices=ids,scope='Single-frame isolated segments, not fabricated video timestamps'),path)
            sealed.append(dict(domain=domain,sha256=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()))
            allout[domain]=outputs
        (folder/'freeze.json').write_text(json.dumps(dict(complete=True,sealed=sealed,GT_free=True,temporal_accuracy_scope=False,detector_scope=False),indent=2))
    finally:projection.project_fisheye624=old;candidates.project_fisheye624=oldcandidate
    # All generated 3D states and fitted results are frozen before labels load.
    starget=torch.load(RUN/'surgical_pose/targets.pt',weights_only=False);sxy=starget['xy'][sdev];valid=starget['valid'][sdev];visible=starget['visible'][sdev]
    roi=sin['inputs']['roi'][sdev,8];scale=(roi[:,2:]-roi[:,:2]).mean(-1);cam=sin['inputs']['camera_params'][sdev]
    error={name:(pinhole_project(v['xyz'],cam)/1408-sxy).norm(dim=-1)/scale[:,None] for name,v in allout['surgery'].items()}
    summary={}
    for name,e in error.items():summary[name]={label:dict(points=int(mask.sum()),PCK10=float((e[mask]<=.1).float().mean()),normalized_error=float(e[mask].mean())) for label,mask in [('all',valid),('visible',valid&visible),('occluded',valid&~visible)]}
    good=valid&(error['reference']<=.05);bad=valid&(error['reference']>.1);groups=collections.defaultdict(list)
    for i,index in enumerate(sdev):groups[sin['metadata'][index]['group']].append(i)
    differences=[float((error['candidate'][ids]-error['reference'][ids])[valid[ids]].mean()) for ids in groups.values()];rng=np.random.default_rng(2026100757)
    ci=np.quantile(np.mean(rng.choice(differences,(2000,len(differences)),replace=True),axis=1),[.025,.975]).tolist()
    nt=torch.load(SOURCE/'targets.pt',weights_only=False,mmap=True);gt=nt['gt'][nids,8];nv=nt['valid'][nids,8]
    native_metrics=metrics(allout['native']['candidate']['xyz'],allout['native']['reference']['xyz'],gt,nv)
    summary.update(native_3D=native_metrics,surgery_good_points=int(good.sum()),surgery_good_harmed=int((good&(error['candidate']>.1)).sum()),surgery_bad_points=int(bad.sum()),surgery_bad_recovered=int((bad&(error['candidate']<=.1)).sum()),surgery_group_CI95=ci,scope='Matched complete parameter sampling, candidate selection and FK fit under human ROIs; no detector metric or temporal validation',default_changed=False)
    (folder/'evaluation.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
