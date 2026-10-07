"""Frozen-weight +/-1.6s versus the completed +/-4s v43 development trial."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from hand3d_v8_common import V7,save
from hand3d_data_v7 import batch,risk_features
from hand3d_risk_v7 import Risk3D
from hand3d_rollout_v8 import project_fisheye624
from parameter_codec_v31 import OUT,observation_batch
from parameter_candidates_v43 import load_observations,select_sequence
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from semantic_parameter_model_v36 import SemanticParameterHand
from stability_trajectory_v43 import fit_stable
from stability_trajectory_v42 import DEFAULT,saved_check
from evaluate_matched_stability_v43 import compare
from calibrate_hand3d_v8 import paired_ci
from prepare_joint_mano_v28 import records_bank
from temporal_window_v44 import ObservationSampler,SHORT_OFFSETS,replace_windows,window_statistics


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--kind',choices=['dit','regression'],default='dit');a=ap.parse_args()
    torch.set_num_threads(4);start=time.time();root=V7.parent
    out=root/'short_context_v44'/a.kind;out.mkdir(parents=True,exist_ok=True)
    if (out/'summary.json').exists():raise RuntimeError('Completed trial is immutable')
    rows,ids,allrows,cache=load_observations(a.device)
    data=torch.load(root/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    primary,full=records_bank(data);sampler=ObservationSampler(primary,full)
    centers=torch.tensor([r['center_fid'] for r in rows])
    f,dt=sampler.sample(centers.tolist())
    assert torch.equal(f[:,8],centers)
    assert dt.abs()[f>0].max()<=1.600003
    inputs=torch.load(root/'joint_mano_v28/inputs.pt',weights_only=False,mmap=True)
    long_f=inputs['feature_ids'][ids];long_dt=inputs['dt'][ids]
    checks=dict(center_fids_exact=True,sampler_gt_free=True,
                long_context=window_statistics(long_f,long_dt),short_context=window_statistics(f,dt))
    # Camera/RGB/XYZ banks and all center observations remain unchanged.
    params=torch.load(root/'aligned_density_v13/camera_params.pt',weights_only=False,mmap=True)
    xy=project_fisheye624(data['xyz_camera_bank'],params)/1408
    observed=torch.isfinite(xy).all(-1)&(xy>=0).all(-1)&(xy<1).all(-1)&data['available_bank']
    xy=torch.nan_to_num(xy)
    infer={k:v for k,v in data.items() if k not in ['gt','valid','gt_uv','uv_valid','original_base_for_evaluation']}
    longinfer=dict(infer,**{k:v[ids] for k,v in inputs.items() if k!='center_ids'})
    shortinfer=replace_windows(infer,f,dt,xy,observed)
    shortinfer['xy'][:,8]=longinfer['xy'][:,8];shortinfer['observed_2d'][:,8]=longinfer['observed_2d'][:,8]
    shortinfer={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in shortinfer.items()}
    longinfer={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in longinfer.items()}
    original_side=torch.load(OUT/'predicted_right_bank.pt',weights_only=False).to(a.device)
    coarse=torch.load(root/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True).to(a.device)
    bank=torch.load(root/'side_data_v16/consensus/native_bank.pt',weights_only=False,mmap=True).to(a.device)
    rc=torch.load(root/'side_data_v16/consensus/risk_dense/risk_all.pt',weights_only=False,map_location=a.device)
    risk=Risk3D(rc['dim']).to(a.device).eval();risk.load_state_dict(rc['model'])
    temp=torch.tensor(json.loads((root/'side_data_v16/consensus/risk_dense/calibration.json').read_text())['temperature'],device=a.device)
    ckpath=root/'matched_parameter_v43'/a.kind/'best.pt'
    ck=torch.load(ckpath,weights_only=False,map_location=a.device)
    model=SemanticParameterHand(ck['kind'],a.device).to(a.device).eval();model.load_state_dict(ck['model'])
    samplepath=out/'candidates.pt'
    if samplepath.exists():samples=torch.load(samplepath,weights_only=False)
    else:
        pieces={k:[] for k in ['states','xyz','right','mean_xyz','mean_state','heat','risk']}
        with torch.inference_mode():
            # Check old-condition lineage before changing context sampling.
            ix=torch.arange(8,device=a.device)
            b=observation_batch(longinfer,ix,cache['risk'],coarse,original_side,preserve_fitted=True)
            b['rgb_native']=bank[longinfer['feature_ids'][ix]]
            with torch.autocast('cuda',dtype=torch.bfloat16):
                enc=model.encode(b);delta=model.sample_trajectory(b,enc,10,4,202610131)
            old=torch.load(root/'matched_stability_v43'/a.kind/'candidates.pt',weights_only=False)
            state=(b['kinematic_coarse'][None]+delta)[:,:,8].float().transpose(0,1).cpu()
            difference=(state-old['states'][:8]).abs().max()
            checks['long_candidate_parameter_max_difference']=float(difference)
            assert float(difference)==0.,'Baseline generator no longer matches completed trial'
            # Risk is a trained, frozen observation model, evaluated on the new
            # window so it does not retain hidden +/-4s conditioning.
            longrisk=(risk(risk_features(batch(longinfer,ix)))/temp).sigmoid()
            checks['long_risk_max_difference']=float((longrisk-cache['risk'][ix]).abs().max())
            for begin in range(0,len(rows),8):
                ix=torch.arange(begin,min(begin+8,len(rows)),device=a.device)
                prob=(risk(risk_features(batch(shortinfer,ix)))/temp).sigmoid()
                b=observation_batch(shortinfer,ix,None,coarse,original_side,preserve_fitted=True)
                b['risk_camera']=prob[:,:,0];b['risk_relative']=prob[:,:,1]
                b['rgb_native']=bank[shortinfer['feature_ids'][ix]]
                def predict(bb):
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        enc=model.encode(bb);delta=model.sample_trajectory(bb,enc,10,4,202610131+begin)
                        states=bb['kinematic_coarse'][None]+delta;side=enc[3]['side_logits'].argmax(-1)
                        xyz=torch.stack([model.codec.decode(s,side)[:,8] for s in states])
                        mean=states.mean(0);meanxyz=model.codec.decode(mean,side)[:,8]
                    return states.float(),xyz.float(),mean.float(),meanxyz.float(),side,enc[3]
                states,xyz,mean,meanxyz,side,h=predict(b)
                if begin==0:
                    poisoned=dict(b,gt=torch.full((len(ix),17,20,3),float('nan'),device=a.device),gt_right=1-side,
                                  gt_shape=torch.ones(len(ix),10,device=a.device)*999)
                    pp=predict(poisoned);assert torch.equal(states,pp[0]) and torch.equal(xyz,pp[1])
                    checks['generator_gt_poison_exact']=True
                pieces['states'].append(states[:,:,8].transpose(0,1).cpu());pieces['xyz'].append(xyz.transpose(0,1).cpu())
                pieces['right'].append(side.cpu());pieces['mean_xyz'].append(meanxyz.cpu());pieces['mean_state'].append(mean[:,8].cpu())
                pieces['heat'].append(h['xy'][:,8].float().cpu());pieces['risk'].append(prob.cpu())
                if begin%400==0:print(json.dumps(dict(stage='sample',kind=a.kind,done=min(begin+8,len(rows)),total=len(rows),seconds=time.time()-start)),flush=True)
        samples=dict(states=torch.cat(pieces['states']),xyz=torch.cat(pieces['xyz']),right=torch.cat(pieces['right']),
                     mean_observation=dict(xyz=torch.cat(pieces['mean_xyz']),parameter_state=torch.cat(pieces['mean_state']),right=torch.cat(pieces['right'])),
                     heat_xy=torch.cat(pieces['heat']),risk=torch.cat(pieces['risk']),checks=checks)
        torch.save(samples,samplepath)
    checks=samples['checks'];cache=dict(cache,risk=samples['risk'].to(a.device))
    # Keep the same per-frame RGB localization evidence in the selector/solver.
    # That localization head has no temporal attention in encode().
    observations={'mean':samples['mean_observation']}
    if a.kind=='dit':
        sequence,selection=select_sequence(samples,cache,rows)
        poisoned=dict(cache,gt=torch.full_like(cache['base'],float('nan')),valid=torch.zeros(len(rows),20,device=a.device,dtype=torch.bool))
        other,ps=select_sequence(samples,poisoned,rows)
        assert torch.equal(sequence['parameter_state'],other['parameter_state']) and selection['selected_index']==ps['selected_index']
        checks['selector_gt_poison_exact']=True;observations['sequence']=sequence;save(out/'sequence_selection.json',selection)
    decoder=ParameterTrajectoryDecoder(a.device);predictions={};constraint_checks={}
    for name,observation in observations.items():
        resultpath=out/(name+'.pt')
        if resultpath.exists():saved=torch.load(resultpath,weights_only=False)
        else:
            observation=dict(observation,parameter_side_outlier=observation['right']!=original_side[centers.to(a.device)].cpu())
            oo={k:v.to(a.device) for k,v in observation.items()}
            result=fit_stable(decoder,cache,oo,rows,DEFAULT,progress=lambda x:print(json.dumps(dict(stage='solver',kind=a.kind,variant=name,**x)),flush=True))
            saved={k:({kk:vv.cpu() for kk,vv in v.items()} if k=='parameters' else v.cpu() if torch.is_tensor(v) else v) for k,v in result.items()}
            torch.save(saved,resultpath)
        pp={k:v.to(a.device) for k,v in saved['parameters'].items()}
        world,check=saved_check(decoder,pp,saved['layout'],DEFAULT)
        assert check['passed'] and torch.allclose(world.cpu(),saved['world'],atol=1e-7,rtol=0)
        source=torch.tensor(saved['layout']['source_map'],device=a.device)
        camera=torch.einsum('njc,nck->njk',world[source]-cache['translation'][:,None],cache['rotation'])
        assert torch.allclose(camera.cpu(),saved['prediction'],atol=1e-7,rtol=0)
        constraint_checks[name]=check;predictions['short_'+name]=saved['prediction'].to(a.device)
    save(out/'freeze.json',dict(outputs_frozen_before_evaluation=True,sha256={n:hashlib.sha256((out/(n+'.pt')).read_bytes()).hexdigest() for n in observations}))
    longdir=root/'matched_stability_v43'/(a.kind+'_side_consistent')
    for name in observations:predictions['long_'+name]=torch.load(longdir/(name+'.pt'),weights_only=False)['prediction'].to(a.device)
    predictions['v42']=torch.load(longdir/'reference_v42.pt',weights_only=False)['prediction'].to(a.device)
    reports,diagnostics,tensors=compare(predictions,cache,rows)
    # Labels become visible only now, after every prediction was saved.
    labels=torch.load(root/'joint_mano_v28/evaluation_labels.pt',weights_only=False,mmap=True)
    gt,valid=labels['gt'][ids].to(a.device),labels['valid'][ids].to(a.device)
    central=torch.tensor([r['window_index'] is not None for r in rows],device=a.device);cr=[r for r in rows if r['window_index'] is not None]
    comparisons={};mask=valid.clone();mask[:,5]=False
    error=lambda p:((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000
    for variant in observations:
        short,long=predictions['short_'+variant],predictions['long_'+variant]
        se,le=error(short),error(long);good=mask&(le<=10);bad=mask&(le>20)
        original=cache['base'].clone()
        for i,r in enumerate(rows):
            if r['window_index'] is not None:original[i]=data['original_base_for_evaluation'][r['window_index']].to(a.device)
        oe=error(original);hard=central&(((oe*mask).sum(-1)/mask.sum(-1).clamp_min(1))>40)
        hm=mask&hard[:,None]&(oe>20)
        comparisons[variant]=dict(paired=paired_ci(short[central],long[central],gt[central],valid[central],cr),
            long_good_points=int(good.sum()),long_good_to_short_over20=int((good&(se>20)).sum()),
            long_bad_points=int(bad.sum()),long_bad_to_short_at_most20=int((bad&(se<=20)).sum()),
            hard_windows=int(hard.sum()),hard_bad_points=int(hm.sum()),hard_recovered_long=int((hm&(le<=20)).sum()),hard_recovered_short=int((hm&(se<=20)).sum()))
    summary=dict(complete=True,kind=a.kind,trial='frozen-weight context resampling',checkpoint=str(ckpath),selected_step=ck['step'],
        checkpoint_sha256=hashlib.sha256(ckpath.read_bytes()).hexdigest(),offsets_frames=SHORT_OFFSETS,seconds=time.time()-start,
        reports=reports,diagnostics=diagnostics,comparisons=comparisons,constraint_checks=constraint_checks,checks=checks,
        risk='Same frozen risk network/temperature, recomputed on short window; no hidden long-window features',
        selector='Same frozen per-frame RGB heatmaps, same costs and v42 limits',
        scope='2673 reused development observations,352centerwindows,4P0003sequences; weights trained with +/-4s, not retrained for1.6s',
        default_changed=False)
    save(out/'summary.json',summary);torch.save(tensors,out/'audit_tensors.pt');save(out/'rows.json',rows)
    print(json.dumps(dict(complete=True,kind=a.kind,seconds=time.time()-start,
        table={k:dict(camera=v['metrics']['camera_mm'],relative=v['metrics']['relative_mm'],coherence=v['coherence']) for k,v in reports.items()},comparisons=comparisons)),flush=True)


if __name__=='__main__':main()
