"""Posthoc dense continuity/anatomy audit; frozen v16, no training/selection.

All observed boxes become centers; GT never chooses inputs, tracks, or side.
World alignment removes camera motion. GT identity is audit metadata only.
"""
import collections, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from hand3d_v8_common import V7
from hand3d_data_v7 import batch, risk_features
from hand3d_risk_v7 import Risk3D
from density_model_v13 import DensityTrajectoryHand3D
from adaptive_projection_v14 import apply
import spatial_rgb_common as s

ROOT = V7.parent
FRESH = ROOT / 'fifth_dense_v16'
MODEL = ROOT / 'side_native_v16'
OUT = ROOT / 'coherence_audit_v27'
CHAINS = [[5,6,7,0],[5,8,9,10,1],[5,11,12,13,2],[5,14,15,16,3],[5,17,18,19,4]]
EDGES = [(u,v) for c in CHAINS for u,v in zip(c,c[1:])]

def summary(x):
    x = x.float().flatten()
    if not len(x): return {'count': 0}
    return dict(count=len(x), mean=float(x.mean()), p50=float(x.quantile(.5)),
                p95=float(x.quantile(.95)), p99=float(x.quantile(.99)), max=float(x.max()))

def prepare():
    data = torch.load(FRESH/'consensus_data.pt', weights_only=False, mmap=True)
    records = json.loads((FRESH/'fresh_rows.json').read_text())
    groups = collections.defaultdict(list)
    for fid, rec in enumerate(records, 1):
        groups[(rec['sequence'],rec['clip'],rec['track_id'])].append((int(rec['timestamp_ns']),fid))
    slots = {}; times = {}
    offsets = np.array([-120,-80,-50,-30,-10,-3,-2,-1,0,1,2,3,10,30,50,80,120])/30.
    for values in groups.values():
        values.sort(); stamps=np.array([x[0] for x in values],np.int64); fids=np.array([x[1] for x in values])
        for stamp,fid in values:
            targets=stamp+np.rint(offsets*1e9).astype(np.int64)
            dist=np.abs(targets[:,None]-stamps[None]); nearest=dist.argmin(1)
            slot=np.where(dist[np.arange(17),nearest]<=20_000_000,fids[nearest],0);slot[8]=fid
            dt=np.where(slot>0,(stamps[nearest]-stamp)/1e9,offsets);dt[8]=0.
            slots[fid]=slot;times[fid]=dt
    # Retain every old evaluated window exactly; extra centers use the same
    # prediction-only sampling rule. Old outputs remain separately auditable.
    for n,fid in enumerate(data['feature_ids'][:,8].tolist()):
        slots[fid]=data['feature_ids'][n].numpy();times[fid]=data['dt'][n].numpy()
    ids=torch.tensor(np.array([slots[k] for k in range(1,len(records)+1)]))
    dt=torch.tensor(np.array([times[k] for k in range(1,len(records)+1)]),dtype=torch.float32)
    uv=torch.zeros(len(records)+1,20,2);visible=torch.zeros(len(records)+1,20,dtype=torch.bool)
    covered=torch.zeros(len(records)+1,dtype=torch.bool)
    uv[data['feature_ids']]=data['xy'];visible[data['feature_ids']]=data['observed_2d'];covered[data['feature_ids']]=True
    assert torch.equal(uv[data['feature_ids']],data['xy'])
    assert torch.equal(visible[data['feature_ids']],data['observed_2d'])
    for fid in torch.nonzero(~covered).flatten().tolist():
        if not fid:continue
        rec=records[fid-1]
        pixels=s.common.from_json(rec['camera']).eye_to_window(data['xyz_camera_bank'][fid].numpy())/1408.
        uv[fid]=torch.tensor(np.nan_to_num(pixels));visible[fid]=torch.tensor(np.isfinite(pixels).all(-1)&(pixels>=0).all(-1)&(pixels<1).all(-1))&data['available_bank'][fid]
    new=dict(data);new.update(feature_ids=ids,dt=dt,xy=uv[ids],observed_2d=visible[ids])
    # Deliberately poisoned target arrays: the inference whitelist ignores them.
    new.update(gt=torch.full((len(records),20,3),float('nan')),valid=torch.zeros(len(records),20,dtype=torch.bool),gt_uv=torch.full((len(records),20,2),float('nan')),uv_valid=torch.zeros(len(records),20,dtype=torch.bool))
    return new, records

@torch.inference_mode()
def predict():
    started=time.time();torch.set_num_threads(4);device='cuda:0';OUT.mkdir(exist_ok=True)
    frozen=json.loads((MODEL/'fifth_seal.json').read_text())
    for name,h in frozen['code_sha256'].items():
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()==h,name
    data,records=prepare();data={k:v.to(device) if torch.is_tensor(v) else v for k,v in data.items()}
    ck=torch.load(ROOT/'side_data_v16/consensus/risk_dense/risk_all.pt',weights_only=False,map_location=device)
    risk=Risk3D(ck['dim']).to(device).eval();risk.load_state_dict(ck['model'])
    temp=torch.tensor(json.loads((ROOT/'side_data_v16/consensus/risk_dense/calibration.json').read_text())['temperature'],device=device)
    ids=torch.arange(len(records),device=device)
    prob=torch.cat([(risk(risk_features(batch(data,ids[start:start+64])))/temp).sigmoid() for start in range(0,len(ids),64)])
    ck=torch.load(MODEL/'consensus/uniform_adaptive/best.pt',weights_only=False,map_location=device)
    model=DensityTrajectoryHand3D('dit',True).to(device).eval();model.load_state_dict(ck['model'])
    bank=torch.load(FRESH/'fresh_dense.pt',weights_only=False,mmap=True).to(device)
    result=collections.defaultdict(list)
    for start in range(0,len(ids),8):
        b=batch(data,ids[start:start+8],prob);b['rgb_native']=bank[data['feature_ids'][start:start+8]]
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_rollout(b,seed=202610114+start)
        raw=p['xyz_camera_m'].float()
        result['prediction'].append(apply(raw,b,frozen['policy']).cpu());result['proposal'].append(raw.cpu())
        if start%400==0:print(json.dumps(dict(completed=min(start+8,len(ids)),total=len(ids),seconds=time.time()-started)),flush=True)
    result={k:torch.cat(v) for k,v in result.items()}
    result['feature_ids']=data['feature_ids'].cpu();result['dt']=data['dt'].cpu()
    result['seed_base']=202610114;result['batch_size']=8;result['seconds']=time.time()-started
    torch.save(result,OUT/'dense_predictions.pt');print('Dense inference complete',flush=True)

def audit():
    torch.set_num_threads(4);d=torch.load(FRESH/'consensus_data.pt',weights_only=False,mmap=True)
    original=torch.load(FRESH/'dense_data.pt',weights_only=False,mmap=True)
    p=torch.load(OUT/'dense_predictions.pt',weights_only=False,mmap=True)
    rec=json.loads((FRESH/'fresh_rows.json').read_text());n=len(rec)
    gt=torch.zeros(n,20,3);valid=torch.zeros(n,20,dtype=torch.bool);side=[];annotations={}
    for k,r in enumerate(rec):
        identity=None
        if r['matched']:
            label=torch.tensor(r['gt']);mask=torch.isfinite(label).all(-1);gt[k]=torch.nan_to_num(label);valid[k]=mask
            sp,sequence,clip=Path(r['image']).relative_to(s.common.ROOT/'export/images').parts[:3]
            path=s.common.ROOT/'export/annotations'/sp/sequence/(clip+'.jsonl')
            if path not in annotations:annotations[path]=[json.loads(x) for x in path.read_text().splitlines()]
            for hand in annotations[path][r['frame']]['hands']:
                xyz=torch.tensor(hand['xyz_camera_m'])
                if mask.all() and torch.isfinite(xyz).all() and (xyz-label).abs().max()<2e-6:
                    assert identity is None;identity=hand['side']
        side.append(identity)
    R=d['rotation'][1:];T=d['translation'][1:]
    def world(x):return torch.einsum('njc,nkc->njk',x,R)+T[:,None]
    gw=world(gt);gr=gw-gw[:,5:6]
    variants={'original_wilor':original['xyz_camera_bank'][1:],'side_corrected_wilor':d['xyz_camera_bank'][1:],'v16':p['prediction'],'raw_dit':p['proposal']}
    i,j=zip(*EDGES);ev=valid[:,i]&valid[:,j];gl=(gt[:,i]-gt[:,j]).norm(dim=-1)*1000
    groups=collections.defaultdict(list)
    for k,r in enumerate(rec):groups[(r['sequence'],r['clip'],r['track_id'])].append(k)
    pairs=[];switches=[];gaps=collections.Counter()
    for ns in groups.values():
        ns.sort(key=lambda k:rec[k]['timestamp_ns'])
        for a,b in zip(ns,ns[1:]):
            gaps[rec[b]['frame']-rec[a]['frame']]+=1
            if rec[b]['frame']-rec[a]['frame']!=1:continue
            if side[a] is None or side[b] is None:continue
            if side[a]!=side[b]:switches.append((a,b));continue
            pairs.append((a,b))
    left,right=map(torch.tensor,zip(*pairs));mask=valid[left]&valid[right];mask[:,5]=False
    emask=ev[left]&ev[right];gd=gr[right]-gr[left];gmove=gd.norm(dim=-1)*1000
    report=dict(scope='Posthoc frozen-v16 audit on previously read fifth12clips, all actual detections as centers. Not new independent performance evidence or selection data. GT used only after inference. Same-GT-hand adjacent frames only; track identity switches reported separately. Diagnostic cutoffs are not learned quality gates.',
        observations=n,valid_frames=int(valid.all(-1).sum()),predicted_tracks=len(groups),frame_gap_counts=dict(gaps),
        adjacent_same_hand_pairs=len(pairs),adjacent_gt_identity_switches=len(switches),inference_seconds=p['seconds'],
        adjacent_dt_seconds=summary(torch.tensor([(rec[b]['timestamp_ns']-rec[a]['timestamp_ns'])/1e9 for a,b in pairs])),metrics={})
    cases={}
    for name,x in variants.items():
        length=(x[:,i]-x[:,j]).norm(dim=-1)*1000;ratio=length/gl.clamp_min(1e-6)
        badbone=((ratio<.5)|(ratio>2))&ev
        wx=world(x);rel=wx-wx[:,5:6];delta=rel[right]-rel[left]
        move=delta.norm(dim=-1)*1000;mis=(delta-gd).norm(dim=-1)*1000
        # Stable-GT counterexamples distinguish spurious jumps from fast motion.
        jump=mask&(gmove<10)&(move>30)
        bonechange=(length[right]-length[left]).abs();rootmis=((wx[right,5]-wx[left,5])-(gw[right,5]-gw[left,5])).norm(dim=-1)*1000
        report['metrics'][name]=dict(bone_length_error_mm=summary((length-gl).abs()[ev]),bone_ratio_to_gt=summary(ratio[ev]),
            bone_extreme_edges=int(badbone.sum()),bone_extreme_frames=int(badbone.any(-1).sum()),bone_edge_count=int(ev.sum()),
            temporal_relative_error_change_mm=summary(mis[mask]),spurious_jump_points=int(jump.sum()),spurious_jump_pairs=int(jump.any(-1).sum()),
            stable_gt_point_pairs=int((mask&(gmove<10)).sum()),bone_length_change_mm=summary(bonechange[emask]),
            bone_change_gt_under1_pred_over10=int(((gl[right]-gl[left]).abs()<1).logical_and(bonechange>10).logical_and(emask).sum()),
            root_temporal_error_change_mm=summary(rootmis))
        tops=[]
        for pairidx,joint in torch.nonzero(jump).tolist():
            a,b=pairs[pairidx];tops.append(dict(sequence=rec[a]['sequence'],clip=rec[a]['clip'],track_id=rec[a]['track_id'],left_frame=rec[a]['frame'],right_frame=rec[b]['frame'],left_index=a,right_index=b,joint=joint,predicted_move_mm=float(move[pairidx,joint]),gt_move_mm=float(gmove[pairidx,joint]),temporal_error_change_mm=float(mis[pairidx,joint]),image_left=rec[a]['image'],image_right=rec[b]['image']))
        tops.sort(key=lambda x:x['predicted_move_mm'],reverse=True)
        bones=[]
        for a,e in torch.nonzero(badbone).tolist():bones.append(dict(sequence=rec[a]['sequence'],clip=rec[a]['clip'],track_id=rec[a]['track_id'],frame=rec[a]['frame'],index=a,edge=EDGES[e],predicted_length_mm=float(length[a,e]),gt_length_mm=float(gl[a,e]),ratio=float(ratio[a,e]),image=rec[a]['image']))
        cases[name]=dict(jump_cases=tops,bone_cases=bones)
    def pair_key(x):return (x['sequence'],x['clip'],x['track_id'],x['left_frame'],x['right_frame'])
    sets={name:{pair_key(x) for x in values['jump_cases']} for name,values in cases.items()}
    base,current=sets['side_corrected_wilor'],sets['v16']
    report['jump_pair_overlap']=dict(inherited=len(base&current),new_vs_corrected_wilor=len(current-base),resolved=len(base-current),baseline_pairs=len(base),v16_pairs=len(current))
    report['score_strata']={}
    for threshold in [0,.1,.25,.5,.75]:
        pairset={(rec[a]['sequence'],rec[a]['clip'],rec[a]['track_id'],rec[a]['frame'],rec[b]['frame']) for a,b in pairs if min(rec[a]['score'],rec[b]['score'])>=threshold}
        eligible=sum(float(x['score'])>=threshold and bool(valid[k].all()) for k,x in enumerate(rec))
        z=dict(eligible_pairs=len(pairset),valid_frames=eligible,variants={})
        for name in cases:
            hits=[x for x in cases[name]['jump_cases'] if pair_key(x) in pairset]
            bones=[x for x in cases[name]['bone_cases'] if rec[x['index']]['score']>=threshold]
            z['variants'][name]=dict(jump_pairs=len({pair_key(x) for x in hits}),jump_points=len(hits),bone_extreme_frames=len({x['index'] for x in bones}),max_jump_case=hits[0] if hits else None)
        report['score_strata'][str(threshold)]=z
    report['diagnostic_definitions']={'spurious_jump':'Wrist-relative world prediction moves >30mm in one frame while same-hand GT moves <10mm; valid finger points only. Does not classify every physical jump.','bone_extreme':'Bone length <0.5 or >2 times same-frame GT. Diagnostic, not complete anatomical validity.','bone_flicker':'Predicted bone length changes >10mm while GT changes <1mm in one frame.'}
    (OUT/'dense_audit.json').write_text(json.dumps(report,indent=2));(OUT/'dense_cases.json').write_text(json.dumps(cases,indent=2))
    print(json.dumps(report,indent=2),flush=True);print('Worst v16 jumps',json.dumps(cases['v16']['jump_cases'][:3],indent=2),flush=True)

if __name__=='__main__':
    if not (OUT/'dense_predictions.pt').exists():predict()
    audit()
