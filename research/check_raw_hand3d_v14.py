"""Engineering checks on raw RGB, no accuracy claims for artificial locks."""
import collections,json,subprocess,sys
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save
from hand3d_data_v7 import batch,risk_features
from encode_hand3d_dense_v14 import encode
from infer_hand3d_v14 import load_models

RUN=V7.parent/'adaptive_projection_v14/dit_dense';FRESH=V7.parent/'fourth_dense_v14'

def main():
    torch.set_num_threads(4);root=Path(__file__).resolve().parent
    def infer(src,dest):
        subprocess.run([sys.executable,str(root/'infer_hand3d_v14.py'),'--input',str(src),'--output',str(dest),'--device','cuda:2'],check=True)
    source=V7.parent/'offline_hand3d_v7_bounded/example_input.json'
    infer(source,RUN/'example_output.json');obj=json.loads(source.read_text());obj['gt']=[999.]*20
    for track in obj['tracks']:
        track['gt_hand_shape']=[-999.]*10
        for f in track['frames']:f.update(gt_xyz=[[999.]*3]*20,gt_side='invented',visibility=[0.]*20,gt_shape=[999.]*10)
    poison=RUN/'poison_input.json';save(poison,obj);infer(poison,RUN/'poison_output.json')
    a=json.loads((RUN/'example_output.json').read_text());p=json.loads((RUN/'poison_output.json').read_text());assert a==p
    missing=json.loads(source.read_text());f=missing['tracks'][0]['frames'][0]
    f['available_3d']=[True]*20;f['available_3d'][1]=False;f['confirmed_3d'][1]=False;f['xyz_camera_m'][1]=[None]*3
    save(RUN/'missing_input.json',missing);infer(RUN/'missing_input.json',RUN/'missing_output.json')
    m=json.loads((RUN/'missing_output.json').read_text())['tracks'][0]['frames'][0]
    assert m['missing_3d_input_review_only'] and not m['automatic_projection_applied'] and m['joints'][1]['xyz_camera_m'] is None
    assert np.isfinite(m['joints'][1]['candidate_xyz_camera_m']).all()
    for j in range(20):
        if j!=1:assert m['joints'][j]['xyz_camera_m']==f['xyz_camera_m'][j]
    # A real 30 FPS track tests the new adapter against the sealed cached path.
    records=json.loads((FRESH/'fresh_rows.json').read_text());groups=collections.defaultdict(list)
    for fid,r in enumerate(records,1):groups[(r['sequence'],r['clip'],r['track_id'])].append((fid,r))
    observations=max(groups.values(),key=len);data=torch.load(FRESH/'dense_data.pt',weights_only=False,mmap=True);native=torch.load(FRESH/'fresh_dense.pt',weights_only=False,mmap=True)
    track=dict(id='engineering_30fps',frames=[dict(image=r['image'],camera=r['camera'],timestamp_s=r['timestamp_ns']*1e-9,box_xyxy=r['box'],box_confidence=r['score'],clip=r['clip'],xyz_camera_m=data['xyz_camera_bank'][fid].tolist(),available_3d=[True]*20,confirmed_3d=[False]*20) for fid,r in observations])
    model,risk,temp,probe,projection,encoder,frozen=load_models('cuda:2')
    b,chosen=encode(track,encoder,probe,projection,'cuda:2');lookup={fid:i for i,(fid,r) in enumerate(observations)}
    windows=[i for i in range(len(data['rows'])) if int(data['feature_ids'][i,8]) in lookup]
    indices=torch.tensor(windows);expected=batch(data,indices);actual={k:v[torch.tensor([lookup[int(data['feature_ids'][i,8])] for i in windows],device='cuda:2')].cpu() for k,v in b.items()}
    expected['rgb_native']=native[data['feature_ids'][indices]]
    for n,i in enumerate(windows):
        seen=chosen[lookup[int(data['feature_ids'][i,8])]]
        sampled=[observations[j][0] if j is not None else 0 for j in seen]
        assert sampled==data['feature_ids'][i].tolist(),(n,sampled,data['feature_ids'][i].tolist())
    parity={}
    for key in ['xyz','available','base','dt','xy','observed_2d','rgb','positions','roi','scores','rays','camera_origin','rgb_valid','confirmed','risk_rgb','rgb_native']:
        difference=(actual[key].float()-expected[key].float()).abs();parity[key]=dict(max=float(difference.max()),mean=float(difference.mean()))
        if key not in ['rgb','risk_rgb','rgb_native']:assert float(difference.max())<1e-4,(key,parity[key])
    x=risk_features(actual);y=risk_features(expected);feature_difference=float((x-y).abs().max())
    with torch.no_grad():
        ar=(risk(x.to('cuda:2'))/temp).sigmoid().cpu();er=(risk(y.to('cuda:2'))/temp).sigmoid().cpu()
    risk_difference=float((ar-er).abs().max())
    save(RUN/'raw_adapter_diagnostic.json',dict(adapter_parity=parity,risk_feature_max_difference=feature_difference,risk_probability_max_difference=risk_difference))
    assert risk_difference<.002,(risk_difference,parity)
    # Repeat the actual sampler with identical inputs, seeds and batch shapes.
    n=min(8,len(windows));ab={k:v[:n].to('cuda:2') for k,v in actual.items()};eb={k:v[:n].to('cuda:2') for k,v in expected.items()}
    for bb,pr in [(ab,ar),(eb,er)]:bb['risk_camera']=pr[:n,:,0].to('cuda:2');bb['risk_relative']=pr[:n,:,1].to('cuda:2')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        ap=model.predict_rollout(ab,seed=202610114)['xyz_camera_m'];ep=model.predict_rollout(eb,seed=202610114)['xyz_camera_m']
    prediction_difference=float((ap.float()-ep.float()).abs().max()*1000);assert prediction_difference<.25
    t=np.array([f['timestamp_s'] for f in track['frames']]);near=float(np.median(np.diff(t)))
    save(RUN/'raw_inference_checks.json',dict(passed=True,GT_field_exact_invariance=True,confirmed_checked=a['confirmed_checked'],missing_input_review_guard_passed=True,partial_locks_use_conservative_policy=True,dense_track_frames=len(observations),dense_windows_compared=len(windows),real_adjacent_interval_s=near,context_ids_exact=True,adapter_parity=parity,risk_feature_max_difference=feature_difference,risk_probability_max_difference=risk_difference,sampler_xyz_max_difference_mm=prediction_difference,scope='Engineering raw/cached parity and lock/missing-field checks; not additional accuracy evidence'))
    save(RUN/'dense_engineering_input.json',dict(image_size=[1408,1408],tracks=[track]));print((RUN/'raw_inference_checks.json').read_text(),flush=True)

if __name__=='__main__':main()
