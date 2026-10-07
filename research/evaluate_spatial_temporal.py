"""Seal before evaluating the reused task holdout; do not tune on these results."""
import json,hashlib,shutil
from pathlib import Path
import spatial_rgb_common as s
import numpy as np,torch
from train_spatial_temporal import load,batch
from spatial_temporal_model import SpatialTemporalCompleter
from evaluate_offline_kp import bootstrap

ARMS=['rgb_dit','rgb_regression','tracks_dit','tracks_regression']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

@torch.inference_mode()
def infer(model,data,ids,gap,mode,ablation=None):
    out={k:[] for k in ['pred','std','linear','mask','all_mask','roi','variant']}
    for start in range(0,len(ids),48):
        ix=ids[start:start+48];variants=1+ix%4 if mode=='partial' else torch.full_like(ix,5 if mode=='full' else 0)
        b,m=batch(data,ix,torch.full_like(ix,gap),variants)
        if ablation=='shuffle_rgb':
            other=ids[(torch.arange(start,start+len(ix),device=ids.device)+len(ids)//2)%len(ids)]
            donor,_=batch(data,other,torch.full_like(ix,gap),variants);b['rgb']=donor['rgb']
        elif ablation=='shuffle_space':b['rgb']=b['rgb'][:,:,torch.arange(191,-1,-1,device=ids.device)]
        elif ablation=='zero_rgb':b['rgb']=torch.zeros_like(b['rgb'])
        with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict(b,samples=4)
        all_m=data['valid'][ix].clone();all_m[:,5]=False
        for k,v in dict(pred=p['xy'],std=p['std'].norm(dim=-1),linear=b['linear'],mask=m,all_mask=all_m,roi=b['roi'][:,8],variant=variants).items():out[k].append(v.cpu().numpy())
    return {k:np.concatenate(v) for k,v in out.items()}

def main():
    torch.set_num_threads(4)
    assert all((s.RUN/a/'done.json').exists() for a in ARMS)
    if (s.RUN/'test_results.json').exists():print('Already evaluated; no reuse');return
    protocol=dict(task='Offline bidirectional RGB + 2D finger keypoint completion',window=17,nominal_hz=6,
        encoder='Full pretrained WiLoR ViT, 32 blocks, 1280 channels, native 16x12 grid; fixed orientation, no handedness input',
        pixels='Identical supplied 1.4x square hand ROI and metadata-seeded native rectangles as v2, with calibrated perspective rectification. Pixels outside ROI zeroed. Mask before encoding.',
        geometry='ROI and camera only; GT and current unmasked handedness excluded from encoder inputs',
        fusion='All 192 cells from every frame enter spatial cross attention; learned joint heatmaps summarize spatial evidence for denoising steps',
        rgb_training='Independent RGB localization first; heatmap auxiliary supervision; 20% whole-window coordinate dropout and 15% joint dropout; no small-gain RGB adapter',
        primary_metric='Artificially occluded finger-point pixel error, gaps1/3/6/9',
        selection='P0003 development only; independent RGB-vs-geometry/content/order checks before temporal fitting',
        evidence_scope='Same v2 held-out task split: P0010/P0015, 8 sequences, 2082 windows; previously opened in older experiments, not untouched data; upstream WiLoR HOT3D overlap unknown',
        limitations=['Supplied hand detections/tracks and camera calibration required; no full hand-box recovery claim',
            'Opaque rectangles are synthetic; natural whole-hand visibility is a proxy, not per-finger occlusion truth',
            'Frozen full visual encoder and learned channel projection; no backbone finetuning',
            '17 sparse frames at 6 Hz; dense 30 fps and temporal jitter not validated'])
    s.save(s.RUN/'protocol.json',protocol)
    seal=s.RUN/'sealed';seal.mkdir(exist_ok=True)
    for arm in ARMS:shutil.copy2(s.RUN/arm/'best.pt',seal/f'{arm}.pt')
    shutil.copy2(s.RUN/'probe_rgb/best.pt',seal/'rgb_probe.pt')
    code=['spatial_rgb_common.py','spatial_rgb_model.py','spatial_temporal_model.py','train_spatial_probe.py',
        'train_spatial_temporal.py','cache_spatial_rgb.py','export_spatial_rgb.py','evaluate_spatial_temporal.py',
        'offline_kp_model.py','pose_residual_dit.py','offline_rgb_encoder.py','wilor_eval_common.py','cache_dit_v3.py','compare_detectors.py','infer_spatial_rgb.py']
    for name in code:shutil.copy2(Path(__file__).parent/name,seal/name)
    s.save(seal/'selection.json',dict(models={a:sha(seal/f'{a}.pt') for a in ARMS},projection_sha256=sha(seal/'rgb_probe.pt'),
        encoder_sha256=sha(s.common.RUN/'assets/wilor_final.mirror.ckpt'),code={n:sha(seal/n) for n in code},
        protocol_sha256=sha(s.RUN/'protocol.json'),test_evaluated=False,steps=10,samples=4,selection='Each checkpoint selected on development only; all four frozen before test inference'))
    rows,data=load('cuda:0');ids=torch.tensor([i for i,r in enumerate(rows) if r['role']=='test'],device='cuda:0')
    dev=torch.tensor([i for i,r in enumerate(rows) if r['role']=='development'],device='cuda:0')
    records,_=s.records_and_index();vis=np.array([records[int(data['feature_ids'][i,8])-1]['visibility_label'] for i in ids.tolist()])
    results={};predictions={};diagnostics={}
    for arm in ARMS:
        ck=torch.load(seal/f'{arm}.pt',map_location='cuda:0',weights_only=False)
        model=SpatialTemporalCompleter(ck['kind'],ck['use_rgb']).to('cuda:0').eval();model.load_state_dict(ck['model'])
        if ck['use_rgb']:
            diag={}
            for ab in [None,'shuffle_rgb','shuffle_space','zero_rgb']:
                o=infer(model,data,dev,6,'partial',ab);e=np.linalg.norm(o['pred']-data['gt'][dev].cpu().numpy(),axis=-1)*1408
                diag[ab or 'normal']=dict(hidden_px=float(e[o['mask']].mean()))
            diagnostics[arm]=diag;s.save(s.RUN/'development_rgb_ablation.json',diagnostics)
        if arm=='rgb_dit':
            scores=[]
            for gap in [1,3,6,9]:
                o=infer(model,data,dev,gap,'partial');e=np.linalg.norm(o['pred']-data['gt'][dev].cpu().numpy(),axis=-1)*1408
                scores.extend((e/(o['std']*1408+5))[o['mask']].tolist())
            s.save(seal/'uncertainty.json',dict(factor=float(np.quantile(scores,.9)),floor_px=5,scope='Development marginal calibration under synthetic partial occlusion only'))
        outs=[];labels=[];gaps=[];clusters=[];indices=[];visibility=[];targets=[]
        for mode in ['partial','full','natural']:
            for gap in ([6] if mode=='natural' else [1,3,6,9]):
                o=infer(model,data,ids,gap,mode);outs.append(o);targets.append(data['gt'][ids].cpu().numpy())
                labels.extend([mode]*len(ids));gaps.extend([gap]*len(ids));clusters.extend(rows[i]['sequence'] for i in ids.tolist());indices.extend(ids.tolist());visibility.extend(vis)
        c={k:np.concatenate([o[k] for o in outs]) for k in outs[0]};gt=np.concatenate(targets)
        labels=np.array(labels);gaps=np.array(gaps);clusters=np.array(clusters);visibility=np.array(visibility)
        e=np.linalg.norm(c['pred']-gt,axis=-1)*1408;base=np.linalg.norm(c['linear']-gt,axis=-1)*1408
        definitions={}
        for mode in ['partial','full']:
            for gap in [0,1,3,6,9]:definitions[f'{mode}/{gap or "all"}']=c['mask']&(labels==mode)[:,None]&((gaps==gap)[:,None] if gap else True)
        definitions['natural/all']=c['all_mask']&(labels=='natural')[:,None]
        definitions['natural/high_hand_occlusion_proxy']=definitions['natural/all']&(visibility<.5)[:,None]
        reports={name:dict(joints=int(m.sum()),mean_px=float(e[m].mean()),linear_px=float(base[m].mean()),
            pck10=float((e[m]<=10).mean()),pck20=float((e[m]<=20).mean()),p95_px=float(np.quantile(e[m],.95)),
            ci95_change_vs_linear_px=bootstrap(e-base,m,clusters)) for name,m in definitions.items() if m.any()}
        results[arm]=dict(arm=arm,step=ck['step'],reports=reports);predictions[arm]=c['pred']
        shared=dict(gt=gt,labels=labels,gaps=gaps,clusters=clusters,visibility=visibility,sample_indices=np.array(indices),mask=c['mask'],all_mask=c['all_mask'],linear=c['linear'],roi=c['roi'],variant=c['variant'])
        if arm=='rgb_dit':
            q=json.loads((seal/'uncertainty.json').read_text());m=definitions['partial/all'];radius=q['factor']*(c['std']*1408+5)
            results[arm]['uncertainty']=dict(partial_coverage=float((e[m]<=radius[m]).mean()),mean_radius_px=float(radius[m].mean()))
        np.savez_compressed(s.RUN/f'{arm}_predictions.npz',predicted=c['pred'],std=c['std'],**shared)
        s.save(s.RUN/f'{arm}_results.json',results[arm]);print(json.dumps(dict(arm=arm,partial=reports['partial/all'],full=reports['full/all'])),flush=True)
        del model;torch.cuda.empty_cache()
    for arm in ['rgb_dit','rgb_regression','tracks_dit','tracks_regression']:
        old=np.load(s.OLD/f'{arm}_predictions.npz')
        for name in ['sample_indices','labels','gaps','mask']:assert np.array_equal(old[name],shared[name]),name
        assert np.allclose(old['gt'],shared['gt'])
        predictions['v2_'+arm]=old['predicted']
    pairs=[('rgb_dit','tracks_dit'),('rgb_regression','tracks_regression'),('rgb_dit','rgb_regression'),
        ('rgb_dit','v2_rgb_dit'),('rgb_regression','v2_rgb_regression'),('rgb_dit','v2_tracks_dit'),('rgb_regression','v2_tracks_regression')]
    comparisons={}
    for a,b in pairs:
        delta=(np.linalg.norm(predictions[a]-shared['gt'],axis=-1)-np.linalg.norm(predictions[b]-shared['gt'],axis=-1))*1408
        comparisons[f'{a}_minus_{b}']={name:dict(change_px=float(delta[m].mean()),ci95=bootstrap(delta,m,shared['clusters'])) for name,m in definitions.items() if m.any()}
    s.save(s.RUN/'test_results.json',dict(complete=True,methods=results,comparisons=comparisons,development_rgb_ablations=diagnostics,
        test_windows=len(ids),source_sequences=len(set(clusters)),scope=protocol['evidence_scope']))

if __name__=='__main__':main()
