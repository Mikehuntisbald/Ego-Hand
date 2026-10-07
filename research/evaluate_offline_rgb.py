import hashlib,json,time,shutil
from pathlib import Path
import wilor_eval_common
import numpy as np,torch
from offline_rgb_data import RUN,save
from offline_rgb_model import RGBKeypointCompleter
from offline_rgb_adapter import RGBAdapterCompleter
from train_offline_rgb import load,batch
from evaluate_offline_kp import bootstrap

ARMS=['rgb_dit','rgb_regression','tracks_dit','tracks_regression']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.inference_mode()
def infer(model,data,ids,gap,mode):
    out={k:[] for k in ['pred','std','linear','mask','all_mask','roi','variant']}
    for start in range(0,len(ids),96):
        ix=ids[start:start+96];n=len(ix)
        variants=1+ix%4 if mode=='partial' else torch.full((n,),5 if mode=='full' else 0,device=ix.device)
        b,mask=batch(data,ix,torch.full((n,),gap,device=ix.device),variants)
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict(b,samples=4)
        all_mask=data['valid'][ix].clone();all_mask[:,5]=False
        for k,v in dict(pred=p['xy'],std=p['std'].norm(dim=-1),linear=b['linear'],mask=mask,
            all_mask=all_mask,roi=b['roi'][:,8],variant=variants).items():out[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}

def main():
    torch.set_num_threads(4)
    while not all((RUN/a/'done.json').exists() for a in ARMS+['adapter_dit','adapter_regression']):time.sleep(10)
    if (RUN/'test_results.json').exists():print('Already evaluated; no test reuse');return
    rows,data=load('cuda:0');ids=torch.tensor([i for i,r in enumerate(rows) if r['role']=='test'],device='cuda:0')
    dev=torch.tensor([i for i,r in enumerate(rows) if r['role']=='development'],device='cuda:0')
    protocol=json.loads((RUN/'protocol.json').read_text());rgb_protocol=json.loads((RUN/'rgb_protocol.json').read_text())
    protocol.update(task='Offline RGB-and-trajectory-conditioned keypoint annotation completion',
        inputs='17-frame masked RGB feature tokens, supplied predicted hand ROIs, surviving 2D observations, masks, and times',
        missing_protocol=rgb_protocol['keypoints'],pixel_protocol=rgb_protocol['pixels'],
        evidence_scope=rgb_protocol['test'],limitations=[rgb_protocol['crop'],rgb_protocol['natural'],
            'Artificial crop-following opaque occluders are not natural object occlusion; crop locations remain given',
            '17 frames at sampled 6 Hz; no dense-30fps or temporal-jitter validation'],
        baselines=['linear interpolation','RGB+tracks regression','RGB+tracks DiT','same-condition tracks-only regression and DiT'])
    save(RUN/'protocol.json',protocol)
    seal=RUN/'sealed';seal.mkdir(exist_ok=True)
    fusion_scores={}
    for mode in ['rgb','adapter']:
        fusion_scores[mode]=sum(torch.load(RUN/f'{mode}_{k}/best.pt',map_location='cpu',weights_only=False)['selection']['hidden_px'] for k in ['dit','regression'])
    fusion=min(fusion_scores,key=fusion_scores.get)
    sources={a:(f'{fusion}_{a.split("_")[1]}' if a.startswith('rgb_') else a) for a in ARMS}
    for arm in ARMS:shutil.copy2(RUN/sources[arm]/'best.pt',seal/f'{arm}.pt')
    code=['offline_kp_model.py','pose_residual_dit.py','offline_rgb_model.py','offline_rgb_adapter.py','offline_rgb_encoder.py','train_offline_rgb.py','train_rgb_adapter.py','evaluate_offline_rgb.py','cache_offline_rgb.py','offline_rgb_data.py','wilor_eval_common.py']
    for name in code:shutil.copy2(Path(__file__).parent/name,seal/name)
    save(seal/'selection.json',dict(models={a:sha(seal/f'{a}.pt') for a in ARMS},code={n:sha(seal/n) for n in code},
        encoder_sha256=rgb_protocol['encoder_sha256'],protocol_sha256=sha(RUN/'protocol.json'),test_evaluated=False,
        selection='Same RGB fusion architecture selected for both methods by sum of their development errors; each checkpoint selected on development only',fusion=fusion,fusion_scores=fusion_scores,sources=sources,steps=10,samples=4))
    source_rows={s:json.loads((wilor_eval_common.ROOT/'experiments'/s/'locked_rows.json').read_text()) for s in {r['source'] for r in rows}}
    visibility=np.array([source_rows[rows[i]['source']][rows[i]['row_index']]['visibility_label'] for i in ids.tolist()])
    results={};predictions={};shared=None
    for arm in ARMS:
        ck=torch.load(seal/f'{arm}.pt',map_location='cuda:0',weights_only=False)
        cls=RGBAdapterCompleter if ck.get('architecture')=='adapter' else RGBKeypointCompleter
        model=cls(ck['kind'],ck['use_rgb']).to('cuda:0').eval();model.load_state_dict(ck['model'])
        if arm=='rgb_dit':
            scores=[]
            for gap in [1,3,6,9]:
                o=infer(model,data,dev,gap,'partial');gt=data['gt'][dev].cpu().numpy()
                e=np.linalg.norm(o['pred']-gt,axis=-1)*1408;scores.extend((e/(o['std']*1408+5))[o['mask']].tolist())
            save(seal/'uncertainty.json',dict(factor=float(np.quantile(scores,.9)),floor_px=5,scope='Empirical partial-synthetic-occlusion development calibration, not a natural-occlusion confidence guarantee'))
        outs=[];labels=[];gaps=[];clusters=[];indices=[];vis=[];targets=[]
        for mode in ['partial','full','natural']:
            for gap in ([6] if mode=='natural' else [1,3,6,9]):
                o=infer(model,data,ids,gap,mode);outs.append(o);targets.append(data['gt'][ids].cpu().numpy())
                labels.extend([mode]*len(ids));gaps.extend([gap]*len(ids));indices.extend(ids.tolist());vis.extend(visibility)
                clusters.extend(rows[i]['sequence'] for i in ids.tolist())
        combined={k:np.concatenate([o[k] for o in outs]) for k in outs[0]};gt=np.concatenate(targets)
        labels=np.array(labels);gaps=np.array(gaps);clusters=np.array(clusters);vis=np.array(vis)
        error=np.linalg.norm(combined['pred']-gt,axis=-1)*1408;base=np.linalg.norm(combined['linear']-gt,axis=-1)*1408
        reports={}
        definitions={}
        for mode in ['partial','full']:
            for gap in [0,1,3,6,9]:
                definitions[f'{mode}/{gap or "all"}']=combined['mask']&(labels==mode)[:,None]&((gaps==gap)[:,None] if gap else True)
        definitions['natural/all']=combined['all_mask']&(labels=='natural')[:,None]
        definitions['natural/high_hand_occlusion_proxy']=definitions['natural/all']&(vis<.5)[:,None]
        for name,mask in definitions.items():
            if not mask.any():reports[name]=dict(joints=0);continue
            reports[name]=dict(joints=int(mask.sum()),mean_px=float(error[mask].mean()),linear_px=float(base[mask].mean()),
                improvement_over_linear_pct=100*(base[mask].mean()-error[mask].mean())/base[mask].mean(),
                pck10=float((error[mask]<=10).mean()),pck20=float((error[mask]<=20).mean()),p95_px=float(np.quantile(error[mask],.95)),
                ci95_change_vs_linear_px=bootstrap(error-base,mask,clusters))
            reports[name]['improvement_over_linear_pct']=float(reports[name]['improvement_over_linear_pct'])
        result=dict(arm=arm,step=ck['step'],use_rgb=ck['use_rgb'],reports=reports)
        if arm=='rgb_dit':
            q=json.loads((seal/'uncertainty.json').read_text());m=definitions['partial/all'];radius=q['factor']*(combined['std']*1408+5)
            result['uncertainty']=dict(partial_test_coverage=float((error[m]<=radius[m]).mean()),mean_radius_px=float(radius[m].mean()))
        results[arm]=result;predictions[arm]=combined['pred'];shared=dict(gt=gt,labels=labels,gaps=gaps,clusters=clusters,visibility=vis,
            sample_indices=np.array(indices),mask=combined['mask'],all_mask=combined['all_mask'],linear=combined['linear'],roi=combined['roi'],variant=combined['variant'])
        np.savez_compressed(RUN/f'{arm}_predictions.npz',predicted=combined['pred'],std=combined['std'],**shared)
        save(RUN/f'{arm}_results.json',result);print(json.dumps(dict(arm=arm,partial=reports['partial/all'],full=reports['full/all'],natural=reports['natural/high_hand_occlusion_proxy'])),flush=True)
        del model;torch.cuda.empty_cache()
    comparisons={}
    for a,b in [('rgb_dit','tracks_dit'),('rgb_regression','tracks_regression'),('rgb_dit','rgb_regression')]:
        ea=np.linalg.norm(predictions[a]-shared['gt'],axis=-1)*1408;eb=np.linalg.norm(predictions[b]-shared['gt'],axis=-1)*1408
        comparisons[f'{a}_minus_{b}']={name:dict(change_px=float((ea-eb)[m].mean()),ci95=bootstrap(ea-eb,m,shared['clusters'])) for name,m in definitions.items() if m.any()}
    save(RUN/'test_results.json',dict(complete=True,methods=results,comparisons=comparisons,
        source_sequences=len(set(clusters)),test_windows=len(ids),scope=protocol['evidence_scope'],
        natural_limit='Natural-image whole-hand-visibility proxy diagnostic; not per-finger natural occlusion ground truth'))

if __name__=='__main__':main()
