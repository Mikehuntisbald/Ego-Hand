"""Create a reproducible example, verify live pixels/cache parity, and package."""
import json,hashlib,shutil,subprocess,sys,tarfile
from pathlib import Path
import spatial_rgb_common as s
import numpy as np,torch

def make_example():
    records,index=s.records_and_index();rows=json.loads((s.OLD/'rows.json').read_text())
    data=torch.load(s.OLD/'windows.pt',weights_only=False)
    candidates=[i for i,r in enumerate(rows) if r['role']=='development' and (index['feature_ids'][i]>0).all()]
    assert candidates
    i=candidates[len(candidates)//2];ids=index['feature_ids'][i];variant=1+i%4;frames=[]
    for t,fid in enumerate(ids.tolist()):
        r=records[fid-1];affected=6<=t<12;obs=data['observed'][i,t].tolist()
        if affected:obs=[False]*20
        frames.append(dict(timestamp_s=float(data['dt'][i,t]),image=r['image'],camera=r['camera'],clip=r['clip'],box_xyxy=r['box'],
            xy_px=[(data['xy'][i,t,j]*1408).tolist() if obs[j] else None for j in range(20)],observed=obs,
            **(dict(occluder_crop_xyxy=index['rectangles'][fid,variant].tolist(),occluder_color=32+64*((r['clip']+1)%4)) if affected else {})))
    obj=dict(image_size=[1408,1408],tracks=[dict(id='development_example',frames=frames)])
    folder=s.RUN/'delivery';folder.mkdir(exist_ok=True);s.save(folder/'example_input.json',obj)
    s.save(folder/'example_context.json',dict(development_window_index=i,central_frame=8,gap=6,variant=variant,gt_in_model_input=False))
    return i,obj

def verify_live(i,obj):
    import wilor_eval_common as common
    from train_spatial_temporal import load,batch
    from offline_kp_model import condition
    from spatial_temporal_model import SpatialTemporalCompleter
    from spatial_rgb_model import SpatialHead
    from infer_spatial_rgb import prepare_track
    device='cuda:0';_,data=load(device)
    full,_=common.load_model(device);encoder=full.backbone;del full
    probe=SpatialHead().to(device).eval();probe.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',map_location=device,weights_only=False)['model'])
    z=prepare_track(obj['tracks'][0],encoder,probe,device)
    ids=torch.tensor([i],device=device);b,_=batch(data,ids,torch.full_like(ids,6),1+ids%4)
    live=condition(torch.tensor(z['xy'][None],device=device),torch.tensor(z['observed'][None],device=device),torch.tensor(z['times'][None],device=device,dtype=torch.float32))
    live.update(rgb=z['bank'][None].to(device),positions=torch.tensor(z['positions'][None],device=device),roi=torch.tensor(z['roi'][None],device=device),rgb_valid=torch.ones(1,17,device=device,dtype=torch.bool))
    feature_error=float((b['rgb'].float()-live['rgb'].float()).abs().max())
    coordinate_input_error=float((b['xy']-live['xy']).abs().max())
    assert coordinate_input_error<1e-6
    checks={}
    for arm in ['rgb_dit','rgb_regression']:
        ck=torch.load(s.RUN/'sealed'/f'{arm}.pt',map_location=device,weights_only=False)
        model=SpatialTemporalCompleter(ck['kind'],True).to(device).eval();model.load_state_dict(ck['model'])
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict(b)['xy'];q=model.predict(live)['xy']
        error=float((p-q).norm(dim=-1).max()*1408);assert error<1.,error
        checks[arm]=dict(live_vs_cached_max_output_difference_px=error)
    del encoder,probe,data,model;torch.cuda.empty_cache()
    return dict(models=checks,live_vs_cached_feature_max_abs=feature_error,hidden_coordinates_absent_in_example=True)

def main():
    torch.set_num_threads(4);assert (s.RUN/'test_results.json').exists()
    code=Path(__file__).resolve().parent;i,obj=make_example();checks=verify_live(i,obj)
    for arm in ['rgb_dit','rgb_regression']:
        subprocess.run([sys.executable,str(code/'infer_spatial_rgb.py'),'--input',str(s.RUN/'delivery/example_input.json'),
            '--output',str(s.RUN/'delivery'/f'example_{arm}.json'),'--arm',arm],cwd=code,check=True)
        out=json.loads((s.RUN/'delivery'/f'example_{arm}.json').read_text());count=0;inferred=0
        for f,o in zip(obj['tracks'][0]['frames'],out['tracks'][0]['frames']):
            for j,seen in enumerate(f['observed']):
                point=o['points'][j]
                if seen:assert point['xy_px']==f['xy_px'][j];count+=1
                else:assert point['source']=='inferred';inferred+=1
                assert point['review_required'] and not point['reviewed']
        checks['models'][arm].update(cli_frames=17,input_coordinates_preserved_exactly=count,inferred_points=inferred)
    s.save(s.RUN/'delivery_checks.json',checks)
    subprocess.run([sys.executable,str(code/'report_spatial_rgb.py')],cwd=code,check=True)
    folder=s.RUN/'delivery';shutil.copytree(s.RUN/'sealed',folder/'sealed',dirs_exist_ok=True)
    shutil.copy2(code/'infer_spatial_rgb.py',folder/'infer_spatial_rgb.py')
    names=['test_results.json','protocol.json','model_checks.json','pixel_checks.json','delivery_checks.json','rgb_localization_gate.json','development_rgb_ablation.json']
    for name in names:shutil.copy2(s.RUN/name,folder/name)
    for arm in ['probe_rgb','probe_geometry','rgb_dit','rgb_regression','tracks_dit','tracks_regression']:
        (folder/'training'/arm).mkdir(parents=True,exist_ok=True)
        for name in ['config.json','history.json','done.json']:shutil.copy2(s.RUN/arm/name,folder/'training'/arm/name)
    sources=folder/'code';sources.mkdir(exist_ok=True)
    for name in ['spatial_rgb_common.py','spatial_rgb_model.py','spatial_temporal_model.py','cache_spatial_rgb.py','train_spatial_probe.py',
        'export_spatial_rgb.py','train_spatial_temporal.py','evaluate_spatial_temporal.py','infer_spatial_rgb.py','check_spatial_pixels.py',
        'check_spatial_models.py','deliver_spatial_rgb.py','report_spatial_rgb.py','run_spatial_stage1.py','run_spatial_stage2.py','run_spatial_experiment.py',
        'offline_kp_model.py','pose_residual_dit.py','wilor_eval_common.py','cache_dit_v3.py','compare_detectors.py','offline_rgb_encoder.py',
        'evaluate_offline_kp.py','offline_kp_data.py','train_offline_kp.py']:
        shutil.copy2(code/name,sources/name)
    manifest={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.rglob('*')) if p.is_file()}
    s.save(folder/'manifest.json',manifest)
    archive=s.RUN/'offline_keypoint_spatial_v3.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(folder,arcname='offline_keypoint_spatial_v3')
    s.save(s.RUN/'delivery_archive.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=len(manifest)))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
