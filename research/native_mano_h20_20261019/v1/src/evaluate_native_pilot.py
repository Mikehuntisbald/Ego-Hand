"""Freeze native predictions first; then score reused dev-calibration labels."""
import hashlib,json,time
from pathlib import Path
import numpy as np,torch
from native_data import Data,ROOT
from native_mano import NativeMANO,common
from native_dit import NativeMANODiT
from train_native import predict

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def stats(x):
    a=np.array(x);return dict(n=int(a.size),mean_mm=float(a.mean()*1000),P95_mm=float(np.quantile(a,.95)*1000),max_mm=float(a.max()*1000))

def main():
    torch.set_num_threads(4);output=ROOT/'evaluation';output.mkdir(exist_ok=False)
    data=Data();codec=NativeMANO();ids=data.audit
    predicted={};hashes={};consistency={}
    for arm in ['regression','dit','protected']:
        checkpoint=ROOT/'runs'/arm/'best_development.pt';ck=torch.load(checkpoint,weights_only=False,map_location='cuda');model=NativeMANODiT(**ck['config']).cuda().eval();model.load_state_dict(ck['model'])
        values=[];deltas=[];zero_rgb=[];parity=[]
        with torch.no_grad():
            for begin in range(0,len(ids),8):
                subset=ids[begin:begin+8];b=data.batch(subset)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    result,delta,_=predict(model,codec,b);off,_,_=predict(model,codec,b,zero_rgb=True)
                exported=codec.export_native(result);back=codec.decode_exported(exported);parity.append(float((back-result['joints']).abs().max()))
                assert torch.equal(result['wrist'],b['state'][:,:3]) and torch.equal(result['betas'],b['betas'])
                values.append(result['joints'].cpu());deltas.append(delta.cpu());zero_rgb.append(off['joints'].cpu())
        predicted[arm]=torch.cat(values);predicted[f'{arm}_RGB_off']=torch.cat(zero_rgb)
        path=output/f'{arm}_predictions.pt';torch.save(dict(indices=ids,joints21=predicted[arm],delta=torch.cat(deltas),RGB_off=predicted[f'{arm}_RGB_off']),path)
        hashes[arm]=dict(checkpoint=sha(checkpoint),predictions=sha(path));consistency[arm]=dict(native_export_redecode_max_m=max(parity),wrist_unchanged=True,shape_unchanged=True)
        # Poison every available loss-only label, recompute fixed-seed inference.
        probe=data.batch(ids[:8])
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p0=predict(model,codec,probe)[0]['joints']
        original=data.labels;data.labels={k:torch.full_like(v,float('nan')) if v.is_floating_point() else torch.zeros_like(v) for k,v in original.items()}
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):p1=predict(model,codec,probe)[0]['joints']
        assert torch.equal(p0,p1);data.labels=original;consistency[arm]['GT_poison_identical']=True
        del model;torch.cuda.empty_cache()
    with torch.no_grad():
        base=[]
        for begin in range(0,len(ids),16):
            b=data.batch(ids[begin:begin+16]);base.append(codec.from_state(b['state'],b['betas'],b['right']).cpu())
        base=torch.cat(base)
    torch.save(dict(indices=ids,joints21=base),output/'coarse_predictions.pt')
    seal=dict(complete=True,unix=time.time(),models=hashes,geometry=consistency,GT_inference=False,
        evaluation_role='reused dev_calibrate clip not used by this pilot optimizer or model selection; historical inspection exists',
        sources={p.name:sha(p) for p in (ROOT/'src').glob('*.py')})
    (output/'predictions_frozen.json').write_text(json.dumps(seal,indent=2))
    # First access to this pilot's calibration labels occurs after prediction freeze.
    source=torch.load('/mnt/why/HOT3D/experiments/online_rgb_3d_v47/targets.pt',weights_only=False,mmap=True)
    source_ids=[data.rows[i]['source_window_index'] for i in ids];gt=source['gt'][source_ids,8].cpu();valid=source['valid'][source_ids,8].cpu()
    rel=lambda x:x-x[:,5:6]
    e0=(base[:,common.MAPPING]-gt).norm(dim=-1);r0=(rel(base[:,common.MAPPING])-rel(gt)).norm(dim=-1)
    metrics={'WiLoR':dict(camera=stats(e0[valid]),relative=stats(r0[valid]))};protection={};rgb_diagnostic={}
    for arm in ['regression','dit','protected']:
        xyz=predicted[arm][:,common.MAPPING];error=(xyz-gt).norm(dim=-1);relative=(rel(xyz)-rel(gt)).norm(dim=-1)
        metrics[arm]=dict(camera=stats(error[valid]),relative=stats(relative[valid]))
        good=valid&(r0<=.01);good[:,5]=False;hard=valid&(r0>.02)
        protection[arm]=dict(base_good10=int(good.sum()),good_harmed_over1mm=int((good&(relative-r0>.001)).sum()),good_to_over20=int((good&(relative>.02)).sum()),
            base_hard20=int(hard.sum()),recovered_to20=int((hard&(relative<=.02)).sum()),improved_over1mm=int((valid&(r0-relative>.001)).sum()),worsened_over1mm=int((valid&(relative-r0>.001)).sum()))
        off=predicted[f'{arm}_RGB_off'][:,common.MAPPING];difference=(xyz-off).norm(dim=-1)
        rgb_diagnostic[arm]=dict(output_difference=stats(difference),RGB_off_relative=stats((rel(off)-rel(gt)).norm(dim=-1)[valid]),
            interpretation='fixed-weight input sensitivity diagnostic, not a retrained RGB ablation')
    groups={}
    for j,i in enumerate(ids):groups.setdefault((data.rows[i]['sequence'],data.rows[i]['clip'],data.rows[i]['track_id']),[]).append(j)
    temporal={}
    for arm in ['WiLoR','regression','dit','protected']:
        p=base[:,common.MAPPING] if arm=='WiLoR' else predicted[arm][:,common.MAPPING];velocity=[];acc=[]
        for _,js in groups.items():
            js.sort(key=lambda j:data.rows[ids[j]]['timestamp_ns']);dt=torch.tensor([(data.rows[ids[b]]['timestamp_ns']-data.rows[ids[a]]['timestamp_ns'])/1e9 for a,b in zip(js[:-1],js[1:])])
            if len(js)<3:continue
            pp=rel(p[js]);gg=rel(gt[js]);pv=torch.diff(pp,dim=0)/(60*dt[:,None,None]);gv=torch.diff(gg,dim=0)/(60*dt[:,None,None]);pair=valid[js[1:]]&valid[js[:-1]]&(dt[:,None]>0)&(dt[:,None]<.2)
            velocity.extend((pv-gv).norm(dim=-1)[pair].tolist());pa=torch.diff(pv,dim=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2);ga=torch.diff(gv,dim=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2)
            triple=pair[1:]&pair[:-1];acc.extend((pa-ga).norm(dim=-1)[triple].tolist())
        temporal[arm]=dict(relative_velocity_error_at60Hz=stats(velocity),relative_acceleration_error_at60Hz2=stats(acc))
    summary=dict(complete=True,training='six train subjects, 304 natural windows; 600 matched updates each',selection='52 dev_select windows; best_development only',audit_windows=len(ids),audit_points=int(valid.sum()),
        metrics=metrics,protection_relative=protection,RGB_sensitivity=rgb_diagnostic,temporal=temporal,geometry=consistency,
        sequence_CI=None,sequence_CI_reason='one reused calibration sequence; insufficient for sequence-level CI',
        native_MANO_parameter_DiT=True,inference_landmark_fitting=False,pseudo_parameter_targets=True,GT_inference=False,safe_default_admitted=False)
    (output/'metrics.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
