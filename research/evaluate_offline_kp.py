"""Freeze checkpoints, calibrate on development, evaluate task holdout once."""
import hashlib,json,time,shutil
from pathlib import Path
import wilor_eval_common
import numpy as np
import torch
from offline_kp_data import RUN,save
from offline_kp_model import KeypointCompleter
from train_offline_kp import load,case

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def bootstrap(delta,mask,clusters):
    pairs=[]
    for c in sorted(set(clusters)):
        m=mask&(clusters==c)[:,None]
        if m.any():pairs.append((float(delta[m].sum()),int(m.sum())))
    if len(pairs)<2:return None
    p=np.asarray(pairs);rng=np.random.default_rng(202610035);idx=rng.integers(len(p),size=(2000,len(p)))
    selected=p[idx];values=selected[:,:,0].sum(-1)/selected[:,:,1].sum(-1)
    return np.quantile(values,[.025,.975]).tolist()

@torch.inference_mode()
def infer(model,data,ids,gap,all_points):
    predictions=[];deviations=[];linear=[];nearest=[];previous=[];masks=[];known=[];original=[];context=[]
    for start in range(0,len(ids),128):
        ix=ids[start:start+128];b,mask=case(data,ix,gap,all_points)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model.predict(b,samples=4)
        predictions.append(out['xy'].cpu().numpy());deviations.append(out['std'].norm(dim=-1).cpu().numpy())
        for dest,key in [(linear,'linear'),(nearest,'nearest'),(previous,'previous')]:dest.append(b[key].cpu().numpy())
        masks.append(mask.cpu().numpy());known.append((~b['missing']).cpu().numpy());original.append(b['xy'][:,8].cpu().numpy());context.append(b['has_context'].cpu().numpy())
    return {k:np.concatenate(v) for k,v in dict(pred=predictions,std=deviations,linear=linear,nearest=nearest,previous=previous,
        mask=masks,known=known,original=original,has_context=context).items()}

def main():
    torch.set_num_threads(4)
    while not all((RUN/k/'done.json').exists() for k in ['dit','regression']):time.sleep(10)
    if (RUN/'test_results.json').exists():print('Task holdout already evaluated; no rerun');return
    rows,data=load('cuda:2');dev=torch.tensor([i for i,r in enumerate(rows) if r['role']=='development'],device='cuda:2')
    test=torch.tensor([i for i,r in enumerate(rows) if r['role']=='test'],device='cuda:2')
    seal=RUN/'sealed';seal.mkdir(exist_ok=True);record={}
    for kind in ['dit','regression']:
        shutil.copy2(RUN/kind/'best.pt',seal/f'{kind}.pt');record[kind]=dict(sha256=digest(seal/f'{kind}.pt'))
    for name in ['offline_kp_model.py','offline_kp_data.py','train_offline_kp.py','evaluate_offline_kp.py','pose_residual_dit.py']:
        shutil.copy2(Path(__file__).parent/name,seal/name)
    save(seal/'selection.json',dict(models=record,code={p.name:digest(p) for p in seal.glob('*.py')},
        protocol_sha256=digest(RUN/'protocol.json'),test_evaluated=False,sampling=dict(steps=10,samples=4),
        checkpoint_selection='Development only; test split never used for model selection'))
    results={};all_predictions={};shared=None
    for kind in ['dit','regression']:
        ck=torch.load(seal/f'{kind}.pt',map_location='cuda:2',weights_only=False)
        model=KeypointCompleter(kind).to('cuda:2').eval();model.load_state_dict(ck['model'])
        scores=[]
        if kind=='dit':
            for gap in [1,3,6,9]:
                for all_points in [False,True]:
                    out=infer(model,data,dev,gap,all_points);gt=data['gt'][dev].cpu().numpy()
                    error=np.linalg.norm(out['pred']-gt,axis=-1)*1408
                    scores.extend((error/(out['std']*1408+5))[out['mask']].tolist())
            radius_factor=float(np.quantile(scores,.9))
            save(seal/'uncertainty.json',dict(radius_factor=radius_factor,std_floor_px=5,development_quantile=.9,
                warning='Empirical marginal development calibration, not a guarantee for correlated video or natural occlusions'))
        outputs=[];stds=[];masks=[];targets=[];lengths=[];types=[];clusters=[];sample_ids=[];baseline={k:[] for k in ['linear','nearest','previous']};known_max=0.;context=[]
        for gap in [1,3,6,9]:
            for all_points in [False,True]:
                out=infer(model,data,test,gap,all_points);gt=data['gt'][test].cpu().numpy()
                outputs.append(out['pred']);stds.append(out['std']);masks.append(out['mask']);targets.append(gt);context.append(out['has_context'])
                if out['known'].any():known_max=max(known_max,float(np.abs(out['pred']-out['original'])[out['known']].max()*1408))
                for key in baseline:baseline[key].append(out[key])
                lengths.extend([gap]*len(test));types.extend(['all_points' if all_points else 'single_finger']*len(test))
                clusters.extend(rows[i]['sequence'] for i in test.tolist());sample_ids.extend(test.tolist())
        p=np.concatenate(outputs);gt=np.concatenate(targets);mask=np.concatenate(masks);std=np.concatenate(stds)*1408
        clusters=np.array(clusters);lengths=np.array(lengths);types=np.array(types);baseline={k:np.concatenate(v) for k,v in baseline.items()}
        errors={kind:np.linalg.norm(p-gt,axis=-1)*1408,**{k:np.linalg.norm(v-gt,axis=-1)*1408 for k,v in baseline.items()}}
        reports={}
        for typ in ['single_finger','all_points']:
            for gap in [0,1,3,6,9]:
                m=mask&(types==typ)[:,None]
                if gap:m=m&(lengths==gap)[:,None]
                e={k:dict(mean_px=float(v[m].mean()),pck10=float((v[m]<=10).mean()),pck20=float((v[m]<=20).mean()),p95_px=float(np.quantile(v[m],.95))) for k,v in errors.items()}
                before=e['linear']['mean_px'];after=e[kind]['mean_px'];ci=bootstrap(errors[kind]-errors['linear'],m,clusters)
                reports[f'{typ}/'+(str(gap) if gap else 'all')]=dict(joints=int(m.sum()),metrics=e,
                    improvement_over_linear_pct=100*(before-after)/max(before,1e-9),ci95_change_px=ci,
                    source_sequences=len(set(clusters[m.any(-1)])))
        primary=reports['single_finger/all'];accepted=known_max==0 and primary['improvement_over_linear_pct']>=5 and primary['ci95_change_px'][1]<0
        result=dict(kind=kind,selected_step=ck['step'],known_input_max_change_px=known_max,
            controlled_gap_improvement_passed=accepted,reports=reports,natural_occlusion_validated=False,
            no_context_fraction=float((mask&~np.concatenate(context)).sum()/mask.sum()))
        if kind=='dit':
            radius=radius_factor*(std+5);result['uncertainty']=dict(radius_factor=radius_factor,test_coverage=float((errors[kind][mask]<=radius[mask]).mean()),mean_radius_px=float(radius[mask].mean()),nominal=.9)
        results[kind]=result;all_predictions[kind]=p
        shared=dict(gt=gt,mask=mask,lengths=lengths,types=types,clusters=clusters,sample_indices=np.array(sample_ids),**baseline)
        np.savez_compressed(RUN/f'{kind}_test_predictions.npz',predicted=p,std_px=std,**shared)
        save(RUN/f'{kind}_test_results.json',result)
        print(json.dumps(dict(kind=kind,primary=primary,known_max=known_max,accepted=accepted)),flush=True)
        del model;torch.cuda.empty_cache()
    de=np.linalg.norm(all_predictions['dit']-shared['gt'],axis=-1)*1408
    re=np.linalg.norm(all_predictions['regression']-shared['gt'],axis=-1)*1408
    m=shared['mask']&(shared['types']=='single_finger')[:,None]
    save(RUN/'test_results.json',dict(complete=True,task='Offline controlled keypoint-gap annotation benchmark',methods=results,
        dit_vs_regression=dict(change_px=float((de-re)[m].mean()),ci95_change_px=bootstrap(de-re,m,shared['clusters'])),
        limitations=json.loads((RUN/'protocol.json').read_text())['limitations'],
        evidence_scope='Task-specific subject-held-out split on previously opened data; synthetic coordinate gaps; no claim of natural full-occlusion recovery'))

if __name__=='__main__':main()
