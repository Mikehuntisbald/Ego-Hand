"""Check runnable-profile parity against the completed fixed-path v45 trial."""
import json
import torch
from hand3d_v8_common import V7,save
from parameter_candidates_v43 import load_observations
from parameter_codec_v31 import OUT
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from complete_hand_tracks_acceleration_v46 import profile_config,solve_profile

torch.set_num_threads(4);root=V7.parent;device='cuda:0';out=root/'acceleration_validation_v46';out.mkdir(exist_ok=True)
rows,_,_,cache=load_observations(device);samples=torch.load(root/'short_context_v44/dit/candidates.pt',weights_only=False)
path=json.loads((root/'short_context_v44/dit/sequence_selection.json').read_text())['selected_index']
ii=torch.arange(len(rows));kk=torch.tensor(path);sides=torch.load(OUT/'predicted_right_bank.pt',weights_only=False)
obs=dict(xyz=samples['xyz'][ii,kk],parameter_state=samples['states'][ii,kk],right=samples['right'])
obs['parameter_side_outlier']=obs['right']!=sides[torch.tensor([r['center_fid'] for r in rows])]
cache=dict(cache,risk=samples['risk'].to(device));decoder=ParameterTrajectoryDecoder(device)
results={}
for profile in ['strict','acc_x2']:
    result=solve_profile(decoder,cache,{k:v.to(device) for k,v in obs.items()},rows,profile=profile)
    old=torch.load(root/'motion_threshold_v45'/(profile+'.pt'),weights_only=False)['prediction']
    delta=float((result['prediction'].cpu()-old).abs().max());assert delta==0.,delta
    assert result['check']['passed'];results[profile]=dict(exact_v45_parity=True,max_difference_m=delta,check=result['check'])
runtime=json.loads((out/'runtime_acc_x2.json').read_text())
assert runtime['motion_profile']=='acc_x2' and runtime['constraint_checks_passed'] and runtime['context_s']==1.6
assert runtime['config']==profile_config('acc_x2')
count=0
for tr in runtime['tracks']:
    assert not tr['manual_anchor_blocked']
    for f in tr['frames']:
        x=torch.tensor(f['xyz_camera_m']);assert x.shape==(20,3) and torch.isfinite(x).all();count+=1
assert count==16
default=json.loads((__import__('pathlib').Path(__file__).parent/'CURRENT_PIPELINE.json').read_text());assert default['version']=='offline_stability_first_v42'
save(out/'runtime_verification.json',dict(passed=True,profiles=results,real_rgb_frames=count,
    root_exported_config_matches_actual_track_limits=True,default_unchanged=True))
print(json.dumps(dict(passed=True,profiles=list(results),runtime_frames=count,default_unchanged=True)))
