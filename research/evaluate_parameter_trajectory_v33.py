"""End-to-end parameter regression -> shared-shape trajectory; dev_select only."""
import json, time, argparse
import torch
from hand3d_v8_common import V7, save
from hand3d_data_v7 import risk_features, batch
from hand3d_risk_v7 import Risk3D
from parameter_codec_v31 import OUT as PARAMETERS, observation_batch
from parameter_temporal_model_v31 import ParameterTemporalHand
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from joint_trajectory_solver_v30 import fit_trajectory
from joint_mano_model_v29 import fuse_overlapping_trajectories
from evaluate_joint_kinematic_v30 import subset, identities, report, eligible, LIMITS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--steps',type=int,default=500)
    ap.add_argument('--name',default='parameter_trajectory_v33')
    ap.add_argument('--model-run',default='parameter_pose_v32')
    ap.add_argument('--own-overlap',action='store_true')
    a=ap.parse_args();torch.set_num_threads(4)
    source=V7.parent/'joint_mano_v28';out=V7.parent/a.name;out.mkdir(exist_ok=True)
    assert not (out/'selection.json').exists()
    allrows=json.loads((source/'rows.json').read_text());n=len(allrows)
    ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select'])
    rows=[allrows[i] for i in ids]
    inputs=torch.load(source/'inputs.pt',weights_only=False,mmap=True)
    cached=torch.load(source/'candidates.pt',weights_only=False,mmap=True)
    fusion=fuse_overlapping_trajectories(cached,inputs,allrows)
    cache=subset(cached,ids,n,a.device);cache['fusion_world']=fusion[ids].to(a.device)
    # Assemble input rows without label arrays. Same frozen feature bank,
    # current RGB crops, timestamps and predicted tracks as the v28 controls.
    data=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    original=data['original_base_for_evaluation']
    infer={k:v for k,v in data.items() if k not in ['gt','valid','gt_uv','uv_valid','original_base_for_evaluation']}
    infer.update({k:v[ids] for k,v in inputs.items() if k!='center_ids'})
    infer={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in infer.items()}
    coarse=torch.load(PARAMETERS/'coarse_state_bank.pt',weights_only=False,mmap=True).to(a.device)
    right=torch.load(PARAMETERS/'predicted_right_bank.pt',weights_only=False).to(a.device)
    bank=torch.load(V7.parent/'side_data_v16/consensus/native_bank.pt',weights_only=False,mmap=True).to(a.device)
    modelrun=V7.parent/a.model_run
    assert (modelrun/'done.json').exists()
    ck=torch.load(modelrun/'best.pt',weights_only=False,map_location=a.device)
    fitted_seed='coarse_seed' in ck['config']
    if fitted_seed:coarse=torch.load(V7.parent/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True).to(a.device)
    if 'query_semantics' in ck['config']:
        from semantic_parameter_model_v36 import SemanticParameterHand
        model=SemanticParameterHand(ck['kind'],a.device).to(a.device).eval()
    else:model=ParameterTemporalHand(ck['kind'],a.device).to(a.device).eval()
    model.load_state_dict(ck['model'])
    save(out/'protocol.json',dict(selection='Only development selection; parameter+frame/wholetrack study',
        generator=str(modelrun),selected_step=ck['step'],steps=a.steps,kind=ck['kind'],own_overlapping_trajectory=a.own_overlap,fitted_seed=fitted_seed,
        safeguards='Original9.5/50mm joint camera+relative point bounds; explicit anatomy; shared shape/chirality; real world motion; wholetrack accept/reject',
        fallback='v16 retained on rejected tracks, included in allmetrics; fallback not anatomically guaranteed',
        manual_locks='Exact manualpoints reject incompatible tracks; no postdecoder XYZ copying',
        default_changed=False,no_artificial_occlusion=True,gt_inference=False))
    predicted=[];states=[];sides=[];trajectories=[];start=time.time()
    for begin in range(0,len(rows),8):
        ix=torch.arange(begin,min(begin+8,len(rows)),device=a.device)
        b=observation_batch(infer,ix,cache['risk'],coarse,right,preserve_fitted=fitted_seed);b['rgb_native']=bank[infer['feature_ids'][ix]]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            p=model.predict_parameters(b,seed=202610131+begin)
        predicted.append(p['xyz_camera_m']);states.append(p['state'][:,8]);sides.append(p['right']);trajectories.append(p['trajectory_xyz_camera_m'])
    obs=dict(xyz=torch.cat(predicted),parameter_state=torch.cat(states),right=torch.cat(sides))
    if a.own_overlap:
        measured=dict(cache,proposal=obs['xyz'],trajectory=torch.cat(trajectories))
        cache['fusion_world']=fuse_overlapping_trajectories(measured,{k:v[ids] for k,v in inputs.items()},rows)
    torch.save({k:v.cpu() for k,v in obs.items()},out/'observations.pt')
    # Labels only enter evaluation after the proposals are completely frozen.
    labels_all=torch.load(source/'evaluation_labels.pt',weights_only=False,mmap=True)
    labels={k:v[ids] for k,v in labels_all.items()};identity=identities(rows,labels)
    original_dense=cache['base'].clone()
    for i,r in enumerate(rows):
        if r['window_index'] is not None:original_dense[i]=original[r['window_index']].to(a.device)
    baseline=report(cache['baseline'],cache,labels,rows,identity,original_dense)
    raw=report(obs['xyz'],cache,labels,rows,identity,original_dense)
    decoder=ParameterTrajectoryDecoder(a.device);check=decoder.reconstruction_check(obs,cache,rows)
    assert check['passed'],check;save(out/'decoder_check.json',check)
    results=[]
    for name,tw,enforce in [('frame',0.,False),('whole_track',.02,True)]:
        config=dict(name=name,steps=a.steps,temporal_weight=tw,overlap_weight=.1 if enforce else 0.,
                    enforce_motion=enforce,rgb_weight=.05,raw_mix=0.,limits=LIMITS,hard_motion_loss=False)
        result=fit_trajectory(decoder,cache,obs,rows,config,progress=lambda x:print(json.dumps(dict(config=name,**x)),flush=True))
        torch.save({k:v.cpu() if torch.is_tensor(v) else ({kk:vv.cpu() for kk,vv in v.items()} if k=='parameters' else v) for k,v in result.items()},out/(name+'.pt'))
        final=report(result['prediction'],cache,labels,rows,identity,original_dense,result['frame_accepted'])
        hypothesis=report(result['hypothesis'],cache,labels,rows,identity,original_dense)
        ok,conditions=eligible(final,baseline)
        entry=dict(config=config,eligible=ok,conditions=conditions,final=final,hypothesis_review_only=hypothesis,
                   accepted_tracks=int(result['accepted'].sum()),tracks=len(result['accepted']))
        results.append(entry);save(out/'partial_results.json',dict(baseline=baseline,generator_review_only=raw,results=results))
        print(json.dumps(entry),flush=True)
    # Test the exact-lock rejection contract on a real short predicted track.
    group=decoder.initialize(obs,cache,rows)['group'];counts=torch.bincount(group)
    g=int(torch.nonzero(counts>=3)[0]);trackids=torch.nonzero(group==g).flatten()
    lc=subset(cache,trackids,len(rows),a.device);lo=subset(obs,trackids,len(rows),a.device);lr=[rows[i] for i in trackids.cpu()]
    lc['confirmed']=torch.zeros(len(lr),20,device=a.device,dtype=torch.bool);lc['confirmed'][0,0]=True
    lc['baseline']=lc['baseline'].clone();lc['baseline'][lc['confirmed']]=lc['base'][lc['confirmed']]
    locked=fit_trajectory(decoder,lc,lo,lr,dict(config,steps=5,enforce_motion=True))
    assert torch.equal(locked['prediction'][lc['confirmed']],lc['base'][lc['confirmed']])
    assert not locked['accepted'].any()
    save(out/'lock_check.json',dict(passed=True,exact_confirmed_preserved=True,incompatible_track_rejected=True))
    passing=[x for x in results if x['eligible'] and x['config']['enforce_motion']]
    save(out/'selection.json',dict(complete=True,baseline=baseline,generator_review_only=raw,results=results,
        approved_on_dev_select=bool(passing),default_changed=False,seconds=time.time()-start,
        scope='Previously used development sequences, no independent fresh assessment. Every fallback included; reviewonly hypothesis is not approved.'))
    print(json.dumps(dict(complete=True,approved=bool(passing),seconds=time.time()-start)),flush=True)


if __name__=='__main__':main()
