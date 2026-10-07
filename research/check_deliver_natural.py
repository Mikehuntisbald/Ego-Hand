import json,shutil,hashlib,subprocess,sys,tarfile
from pathlib import Path
import numpy as np,torch
import spatial_rgb_common as s
from natural_reliability import RUN,RiskHead,risk_features
from infer_natural_reliability import encode_track,windows,predict_windows,load_models
from natural_corrector import natural_batch
from train_spatial_temporal import load

def main():
    torch.set_num_threads(4);code=Path(__file__).resolve().parent;folder=RUN/'delivery';folder.mkdir(exist_ok=True)
    records,index=s.records_and_index();rows,data=load('cuda:0')
    eligible=[i for i,r in enumerate(rows) if r['role']=='development' and (index['feature_ids'][i]>0).all()]
    i=eligible[len(eligible)//2];frames=[]
    for t,fid in enumerate(index['feature_ids'][i].tolist()):
        r=records[fid-1];available=data['observed'][i,t].tolist();confirmed=[False]*20
        # Explicit locks in this fixture check immutability only; they are not
        # used to train/evaluate recovery accuracy or pretend to be human labels.
        for j in [0,5]:confirmed[j]=available[j]
        frames.append(dict(image=r['image'],camera=r['camera'],clip=r['clip'],box_xyxy=r['box'],box_confidence=r['score'],timestamp_s=float(data['dt'][i,t]),
            xy_px=[(data['xy'][i,t,j].cpu()*1408).tolist() if available[j] else None for j in range(20)],available=available,confirmed=confirmed))
    obj=dict(image_size=[1408,1408],tracks=[dict(id='development_integration_fixture',frames=frames)])
    s.save(folder/'example_input.json',obj)
    models,risk,temp,probe,policy,projection=load_models(RUN,'cuda:0')
    full,_=s.common.load_model('cuda:0');encoder=full.backbone;del full
    encoded=encode_track(obj['tracks'][0],encoder,probe,projection,'cuda:0');w=windows(encoded,'cuda:0')
    raw=torch.load(RUN/'features.pt',weights_only=False,mmap=True)
    feature=risk_features(w['xy'][8:9],w['available'][8:9],w['dt'][8:9],w['roi'][8:9],w['scores'][8:9],w['risk_rgb'][8:9],w['positions'][8:9])
    difference=float((feature.cpu()[0]-raw['x'][i]).abs().max());assert difference<1e-3,difference
    dirty=w['xy'].clone();dirty[~w['available']]=123456
    cleanfeat=risk_features(w['xy'],w['available'],w['dt'],w['roi'],w['scores'],w['risk_rgb'],w['positions'])
    dirtyfeat=risk_features(dirty,w['available'],w['dt'],w['roi'],w['scores'],w['risk_rgb'],w['positions'])
    assert torch.equal(cleanfeat,dirtyfeat)
    probability=torch.load(RUN/'probabilities.pt',weights_only=False)['joint'].to('cuda:0');ids=torch.tensor([i],device='cuda:0')
    b=natural_batch(data,ids,probability,w['confirmed'][8:9]);altered=dict(data);altered['gt']=torch.full_like(data['gt'],999)
    b2=natural_batch(altered,ids,probability,w['confirmed'][8:9]);assert all(torch.equal(b[k],b2[k]) for k in b)
    predictions=predict_windows(w,models,risk,temp,probe,policy)
    assert torch.equal(predictions['xy'][torch.from_numpy(encoded['confirmed'])],torch.from_numpy(encoded['xy'])[torch.from_numpy(encoded['confirmed'])])
    cachedprob=probability[i];liveprob=predictions['p_bad'][8].to('cuda:0');risk_difference=float((cachedprob-liveprob).abs().max());assert risk_difference<1e-4,risk_difference
    spatial_error=float((data['bank'][data['feature_ids'][i],0].cpu()-encoded['spatial']).abs().max());assert spatial_error<1e-3,spatial_error
    # Compare candidate outputs with identical batch shape and sampling seed.
    rawpred={}
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        for arm,model in models.items():rawpred[arm]=model.predict(b)['xy'].cpu()
    single={k:v[8:9] for k,v in w.items()};live=predict_windows(single,models,risk,temp,probe,policy)
    errors={arm:float((rawpred[arm]-live[arm]).norm(dim=-1).max()*1408) for arm in models}
    assert max(errors.values())<.1,errors
    checks=dict(live_vs_cached_risk_feature_max_abs=difference,live_vs_cached_probability_max_abs=risk_difference,
        live_vs_cached_spatial_feature_max_abs=spatial_error,candidate_output_max_difference_px=errors,
        unavailable_coordinates_do_not_affect_risk=True,gt_not_used_in_conditioning=True,confirmed_points_preserved_exactly=True,
        fixture_scope='17 natural development frames, no pixel or coordinate masking; confirmed flags are interface test fixtures only')
    del models,risk,probe,encoder,data,raw,w;torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(code/'infer_natural_reliability.py'),'--input',str(folder/'example_input.json'),'--output',str(folder/'example_output.json')],cwd=code,check=True)
    result=json.loads((folder/'example_output.json').read_text());locked=edited=review=0
    for f,o in zip(frames,result['tracks'][0]['frames']):
        for j,p in enumerate(o['points']):
            if f['confirmed'][j]:assert p['xy_px']==f['xy_px'][j] and not p['review_required'];locked+=1
            else:assert p['review_required']
            edited+=p['correction_applied'];review+=p['needs_special_review']
    assert edited>0,'Correction path was never activated'
    checks.update(cli_frames=17,confirmed_json_coordinates_preserved=locked,automatic_corrections=edited,flagged_for_special_review=review)
    s.save(RUN/'delivery_checks.json',checks)
    subprocess.run([sys.executable,str(code/'report_natural_v4.py')],cwd=code,check=True)
    shutil.copytree(RUN/'sealed',folder/'sealed',dirs_exist_ok=True)
    deps=['infer_natural_reliability.py','wilor_eval_common.py','cache_dit_v3.py','compare_detectors.py','offline_rgb_encoder.py']
    for name in deps:shutil.copy2(code/name,folder/'sealed'/name)
    shutil.copy2(code/'infer_natural_reliability.py',folder/'infer_natural_reliability.py')
    s.save(folder/'sealed/manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (folder/'sealed').iterdir() if p.is_file() and p.name!='manifest.json'})
    for name in ['test_results.json','protocol.json','policy.json','risk_calibration.json','delivery_checks.json']:shutil.copy2(RUN/name,folder/name)
    refinement=json.loads((RUN/'policy_refinement.json').read_text());s.save(folder/'policy_refinement_summary.json',{k:v for k,v in refinement.items() if k!='trials'})
    source=folder/'code';source.mkdir(exist_ok=True)
    names=['natural_reliability.py','natural_corrector.py','natural_policy.py','train_natural_corrector.py','evaluate_natural_reliability.py','infer_natural_reliability.py',
        'refine_natural_policy.py','check_deliver_natural.py','report_natural_v4.py','run_natural_v4.py','train_spatial_temporal.py','evaluate_offline_kp.py','offline_kp_data.py','train_offline_kp.py']
    for name in names:shutil.copy2(code/name,source/name)
    for arm in ['dit','regression']:
        (folder/'training'/arm).mkdir(parents=True,exist_ok=True)
        for name in ['config.json','history.json','done.json']:shutil.copy2(RUN/arm/name,folder/'training'/arm/name)
    s.save(folder/'manifest.json',{str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'})
    archive=RUN/'natural_reliability_v4.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:tar.add(folder,arcname='natural_reliability_v4')
    s.save(RUN/'delivery_archive.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
    print(json.dumps(checks,indent=2))

if __name__=='__main__':main()
