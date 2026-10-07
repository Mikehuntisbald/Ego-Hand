"""Raw detector/WiLoR/RGB engineering checks, separately from locked accuracy."""
import argparse,collections,json,subprocess,sys
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save
from hand3d_data_v7 import batch,risk_features
from encode_hand3d_dense_v14 import encode
from infer_hand3d_v16 import load_models
from complete_hand_tracks_v16 import reconstruct
RUN=V7.parent/'side_native_v16';FRESH=V7.parent/'fifth_dense_v16';CODE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser();parser.add_argument('--parity-only',action='store_true');args=parser.parse_args()
    def complete(src,dest,prepared=None):
        if args.parity_only:
            assert dest.exists(),'Reuse requires previously completed raw fixtures'
            return
        cmd=[sys.executable,str(CODE/'complete_hand_tracks_v16.py'),'--input',str(src),'--output',str(dest),'--device','cuda:2']
        if prepared:cmd+=['--prepared-output',str(prepared)]
        subprocess.run(cmd,check=True)
    source=V7.parent/'offline_hand3d_v7_bounded/example_input.json';complete(source,RUN/'example_output.json')
    a=json.loads((RUN/'example_output.json').read_text());assert a['confirmed_checked']==34
    obj=json.loads(source.read_text());obj['gt']=[999.]*20
    for tr in obj['tracks']:
        tr['gt_hand_shape']=[-999.]*10
        for f in tr['frames']:f.update(gt_xyz=[[999.]*3]*20,gt_side='invented',visibility=[0.]*20,gt_shape=[999.]*10)
    save(RUN/'poison_input.json',obj);complete(RUN/'poison_input.json',RUN/'poison_output.json');assert a==json.loads((RUN/'poison_output.json').read_text())
    missing=json.loads(source.read_text());f=missing['tracks'][0]['frames'][0];f['available_3d']=[True]*20;f['available_3d'][1]=False;f['confirmed_3d'][1]=False;f['xyz_camera_m'][1]=[None]*3
    save(RUN/'missing_input.json',missing);complete(RUN/'missing_input.json',RUN/'missing_output.json');out=json.loads((RUN/'missing_output.json').read_text())['tracks'][0]['frames'][0]
    assert out['missing_3d_input_review_only'] and not out['automatic_projection_applied'] and out['joints'][1]['xyz_camera_m'] is None
    assert np.isfinite(out['joints'][1]['candidate_xyz_camera_m']).all()
    for j in range(20):
        if j!=1:assert out['joints'][j]['xyz_camera_m']==f['xyz_camera_m'][j]
    records=json.loads((FRESH/'fresh_rows.json').read_text());groups=collections.defaultdict(list);changes={r['fid'] for r in json.loads((FRESH/'side_choices_prediction_only.json').read_text())}
    for fid,r in enumerate(records,1):groups[(r['sequence'],r['clip'],r['track_id'])].append((fid,r))
    original=torch.load(FRESH/'dense_data.pt',weights_only=False,mmap=True);data=torch.load(FRESH/'consensus_data.pt',weights_only=False,mmap=True)
    center_fids=set(data['feature_ids'][:,8].tolist())
    eligible=[g for g in groups.values() if sum(fid in center_fids for fid,r in g)>=8]
    assert eligible
    obs=max(eligible,key=lambda g:(sum(fid in changes and fid in center_fids for fid,r in g),sum(fid in changes for fid,r in g),len(g)))
    track=dict(id='raw30fps',frames=[dict(image=r['image'],camera=r['camera'],timestamp_s=r['timestamp_ns']*1e-9,box_xyxy=r['box'],box_confidence=r['score'],clip=r['clip'],xyz_camera_m=original['xyz_camera_bank'][fid].tolist(),side_predictions=r['side_predictions'],available_3d=[True]*20,confirmed_3d=[False]*20) for fid,r in obs]);raw_source=dict(image_size=[1408,1408],tracks=[track]);save(RUN/'dense_engineering_input.json',raw_source)
    provided=reconstruct(raw_source,'cuda:2',True);actual=reconstruct(raw_source,'cuda:2',False)
    xyz=torch.tensor([f['xyz_camera_m'] for f in actual['tracks'][0]['frames']]);reference=data['xyz_camera_bank'][torch.tensor([fid for fid,r in obs])]
    pose_diff=float((xyz-reference).abs().max()*1000);assert pose_diff<.25,pose_diff
    assert [f['predicted_right'] for f in actual['tracks'][0]['frames']]==[f['predicted_right'] for f in provided['tracks'][0]['frames']]
    model,risk,temp,probe,projector,encoder,frozen=load_models('cuda:2');b,chosen=encode(actual['tracks'][0],encoder,probe,projector,'cuda:2');lookup={fid:k for k,(fid,r) in enumerate(obs)}
    windows=[i for i in range(len(data['rows'])) if int(data['feature_ids'][i,8]) in lookup];assert windows;ids=torch.tensor(windows,dtype=torch.long);expected=batch(data,ids);native=torch.load(FRESH/'fresh_dense.pt',weights_only=False,mmap=True);expected['rgb_native']=native[data['feature_ids'][ids]]
    actual_b={k:x[torch.tensor([lookup[int(data['feature_ids'][i,8])] for i in windows],device='cuda:2')].cpu() for k,x in b.items()}
    for i in windows:
        picked=chosen[lookup[int(data['feature_ids'][i,8])]];assert [obs[j][0] if j is not None else 0 for j in picked]==data['feature_ids'][i].tolist()
    # Encode produces observations; the risk outputs are computed next. The
    # cached batch's default zero risks are not observations to compare here.
    parity={k:float((actual_b[k].float()-expected[k].float()).abs().max()) for k in expected if k not in ['risk_camera','risk_relative']}
    assert parity['rgb_native']==0 and parity['rgb']==0 and parity['risk_rgb']==0
    with torch.no_grad():
        ar=(risk(risk_features(actual_b).to('cuda:2'))/temp).sigmoid().cpu();er=(risk(risk_features(expected).to('cuda:2'))/temp).sigmoid().cpu()
    pd=float((ar-er).abs().max());assert pd<.002
    n=min(8,len(windows));ab={k:x[:n].to('cuda:2') for k,x in actual_b.items()};eb={k:x[:n].to('cuda:2') for k,x in expected.items()}
    for bb,pr in [(ab,ar),(eb,er)]:bb['risk_camera']=pr[:n,:,0].to('cuda:2');bb['risk_relative']=pr[:n,:,1].to('cuda:2')
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):ap=model.predict_rollout(ab,seed=202610114)['xyz_camera_m'];ep=model.predict_rollout(eb,seed=202610114)['xyz_camera_m']
    pred_diff=float((ap.float()-ep.float()).abs().max()*1000);save(RUN/'raw_parity_diagnostic.json',dict(adapter_parity=parity,upstream_xyz_max_mm=pose_diff,risk_probability_max=pd,sampler_xyz_max_mm=pred_diff));assert pred_diff<.25,pred_diff
    save(RUN/'dense_engineering_prepared.json',actual)
    save(RUN/'raw_inference_checks.json',dict(passed=True,confirmed_checked=34,GT_field_exact_invariance=True,missing_input_review_guard=True,raw_detector_side_choices_match_cached=True,dense_track_frames=len(obs),side_corrected_frames=sum(f['side_corrected'] for f in actual['tracks'][0]['frames']),context_ids_exact=True,upstream_xyz_max_difference_mm=pose_diff,risk_probability_max_difference=pd,sampler_xyz_max_difference_mm=pred_diff,adapter_parity=parity,scope='Raw detector/WiLoR/RGB engineering parity and locks; not additional accuracy evidence'))
    print((RUN/'raw_inference_checks.json').read_text(),flush=True)

if __name__=='__main__':main()
