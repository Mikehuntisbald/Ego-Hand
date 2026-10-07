import argparse,json,hashlib
import numpy as np,torch
from hand3d_v8_common import RUN,V7,load,batch,save,predict,metrics,score
from hand3d_rollout_v8 import RolloutHand3D
from bounded_policy_v8 import apply
from calibrate_hand3d_v8 import paired_ci
from hand3d_visual_v8 import NativeSpatialStem,dense_bank
from hand3d_risk_v7 import Risk3D

def restore(folder,device,data,fresh=False):
    ck=torch.load(RUN/folder/'best.pt',weights_only=False,map_location=device);model=RolloutHand3D(ck['kind'],True,cap_m=ck['config'].get('cap_m')).to(device).eval();model.load_state_dict(ck['model']);visual=None
    if 'visual' in ck:
        if folder.endswith('_tail'):
            from train_hand3d_tail_v8 import LiveTail
            from finetune_visual_v5 import load_visual,prefix_bank
            backbone,stem=load_visual(device);visual=LiveTail(backbone,stem).to(device).eval();del backbone,stem;visual.load_state_dict(ck['visual'])
            visual.prefix=torch.load(RUN/'fresh_prefix.pt',weights_only=False).to(device) if fresh else prefix_bank(device,len(data['world']))
        else:
            bank=torch.load(RUN/'fresh_dense.pt',weights_only=False).to(device) if fresh else dense_bank(device,len(data['world']))
            visual=NativeSpatialStem(device,bank).to(device).eval();visual.load_state_dict(ck['visual'])
    return model,visual,ck

@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['old','fresh','ablation'],default='old');p.add_argument('--device',default='cuda:0');a=p.parse_args();torch.set_num_threads(4)
    selection=json.loads((RUN/'development_selection.json').read_text());assert selection['approved_on_development'];selected=selection['selected'];folder=selected['folder'];policy=selected['policy']
    fresh=a.split=='fresh'
    if fresh:
        seal=json.loads((RUN/'fresh_evaluation_seal.json').read_text());assert seal['folder']==folder and seal['policy']==policy
        assert hashlib.sha256((RUN/folder/'best.pt').read_bytes()).hexdigest()==seal['checkpoint_sha256']
        assert hashlib.sha256((RUN/'fresh_manifest.json').read_bytes()).hexdigest()==seal['manifest_sha256']
        raw=torch.load(RUN/'fresh_data.pt',weights_only=False,mmap=True);data={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in raw.items()};ids=torch.arange(len(data['rows']),device=a.device)
        risk_ck=torch.load(V7/'risk_all.pt',weights_only=False,map_location=a.device);risk=Risk3D(risk_ck['dim']).to(a.device).eval();risk.load_state_dict(risk_ck['model']);temp=torch.tensor(json.loads((V7/'risk_calibration.json').read_text())['temperature'],device=a.device)
        from hand3d_data_v7 import risk_features
        parts=[]
        for start in range(0,len(ids),64):parts.append((risk(risk_features(batch(data,ids[start:start+64])))/temp).sigmoid())
        prob=torch.cat(parts)
    else:
        data=load(a.device);roles=np.asarray(data['roles']);role='dev_calibrate' if a.split=='ablation' else 'test';ids=torch.tensor(np.where(roles==role)[0],device=a.device);prob=torch.load(V7/'risk_probabilities.pt',weights_only=False)['joint'].to(a.device)
    model,visual,ck=restore(folder,a.device,data,fresh);size=4 if folder.startswith('dit_3d_') else 16
    output=predict(model,data,prob,ids,size=size,visual=visual);base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids];valid=data['valid'][ids];rows=[data['rows'][int(i)] for i in ids];pred=apply(output[policy['source']],base,policy)
    final=metrics(pred,base,gt,valid);baseline=metrics(base,base,gt,valid);raw_metrics=metrics(output['raw_xyz_camera_m'],base,gt,valid);ci=paired_ci(pred,base,gt,valid,rows);_,feasible=score(final)
    feasible=feasible and ci['relative']['ci95_delta_mm'][1]<0 and final['relative_bad_recovered20']>0 and final['relative_bad_mean_mm']<baseline['relative_bad_mean_mm']
    groups={}
    if not fresh:
        queue=json.loads((V7.parent/'natural_reliability_v4/delivery/hardcase_review_queue.json').read_text());lookup={int(i):j for j,i in enumerate(ids)};hard=[lookup[q['window_index']] for q in queue if q['window_index'] in lookup];severe=[lookup[q['window_index']] for q in queue if q['window_index'] in lookup and q['after_px']>40]
        for name,ix in [('natural_hardcases_47',hard),('previous_severe_8',severe)]:
            if ix:groups[name]=metrics(pred[ix],base[ix],gt[ix],valid[ix])
    else:
        # Error-based diagnostic is distinct from natural per-finger visibility.
        from hand3d_temporal_v7 import WRIST
        relative=((base-base[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1)*1000;mask=valid.clone();mask[:,WRIST]=False
        hard=((relative*mask).sum(-1)/mask.sum(-1).clamp_min(1)>40)
        if hard.any():groups['baseline_relative_over40mm']=metrics(pred[hard],base[hard],gt[hard],valid[hard])
    report=dict(split=a.split,folder=folder,selected_step=ck['step'],policy=policy,windows=len(ids),sequences=len({r['sequence'] for r in rows}),baseline=baseline,final=final,raw=raw_metrics,paired_ci=ci,groups=groups,passed=feasible,scope='Unused source clips of reused subjects/sequences; not new-subject or pretraining-disjoint evidence' if fresh else 'Previously inspected diagnostic data; no new independent generalization claim')
    save(RUN/(a.split+'_results.json'),report);torch.save(dict(indices=ids.cpu(),prediction=pred.cpu(),output={k:v.cpu() for k,v in output.items()},base=base.cpu(),gt=gt.cpu(),valid=valid.cpu(),rows=rows),RUN/(a.split+'_predictions.pt'));print(json.dumps(report),flush=True)
    if a.split=='ablation':
        tests={};model.eval()
        for mode in ['zero_rgb','shuffle_rgb','center_only']:
            outputs=[]
            for start in range(0,len(ids),size):
                ix=ids[start:start+size];b=batch(data,ix,prob)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    if visual:b=visual.replace_batch(b,data,ix)
                    if mode=='zero_rgb':b['rgb']=torch.zeros_like(b['rgb'])
                    elif mode=='shuffle_rgb':b['rgb']=b['rgb'].flip(0)
                    else:
                        keep=torch.zeros_like(b['rgb_valid']);keep[:,8]=True;b['rgb_valid']=b['rgb_valid']&keep;b['available']=b['available']&keep[...,None]
                    y=model.predict(b,seed=202610081+start)
                outputs.append(y[policy['source']].float())
            corrected=apply(torch.cat(outputs),base,policy);tests[mode]=metrics(corrected,base,gt,valid)
        save(RUN/'conditioning_ablation.json',dict(normal=final,ablations=tests,scope='Development calibration perturbations; no artificial-occlusion recovery claim'))

if __name__=='__main__':main()
