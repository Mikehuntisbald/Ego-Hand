"""Verify completed training, actual trained RGB runtime and saved guarantees."""
import hashlib,json
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import save
from online_parameter_model_v47 import RUN
from complete_hand_tracks_acceleration_v46 import profile_config

def main():
    arm={}
    for mode in ['joint','frozen']:
        folder=RUN/f'dit_{mode}';done=json.loads((folder/'done.json').read_text());assert done['complete'] and done['steps']==600
        arm[mode]=json.loads((folder/'config.json').read_text())
        grad=json.loads((folder/'gradients.json').read_text());update=json.loads((folder/'weight_update.json').read_text());parity=json.loads((folder/'initial_parity.json').read_text())
        assert grad['passed'] and update['passed'] and parity['gt_poison_exact']
        if mode=='joint':assert all(v>0 for v in grad['block_gradient_norms'].values()) and update['last_block_max_weight_change']>0
        else:assert all(v==0 for v in grad['block_gradient_norms'].values()) and update['last_block_max_weight_change']==0
    assert arm['joint']['batch_plan_sha256']==arm['frozen']['batch_plan_sha256']
    assert arm['joint']['initial_checkpoint_sha256']==arm['frozen']['initial_checkpoint_sha256']
    assert json.loads((RUN/'dit_joint/fk_gradient_path.json').read_text())['passed']
    r=json.loads((RUN/'runtime_joint_last.json').read_text());assert r['constraint_checks_passed'] and r['context_s']==1.6
    assert r['config']==profile_config('acc_x2') and r['checkpoint'].endswith('dit_joint/last.pt')
    count=0
    for track in r['tracks']:
        assert track['constraints']['passed'] and not track['manual_anchor_blocked']
        for frame in track['frames']:
            xyz=np.asarray(frame['xyz_camera_m']);assert xyz.shape==(20,3) and np.isfinite(xyz).all();count+=1
    assert count==16
    # Confirm that last.pt actually contains changed visual weights; best.pt
    # may legitimately be the untouched step0 selected by the protection gate.
    last=torch.load(RUN/'dit_joint/last.pt',weights_only=False,map_location='cpu')
    zero=torch.load(RUN/'dit_joint/best.pt',weights_only=False,map_location='cpu')
    assert last['step']==600
    if zero['step']==0:
        delta=max(float((v-zero['visual_tail'][k]).abs().max()) for k,v in last['visual_tail'].items())
        assert delta>0
    else:delta=None
    default=json.loads((Path(__file__).parent/'CURRENT_PIPELINE.json').read_text());assert default['version']=='offline_stability_first_v42'
    evidence=dict(passed=True,both_600step_trainings_complete=True,identical_initialization_and_batch_plan=True,
        pure_fk_3d_gradient_reaches_visual=True,all_last4_visual_layers_update=True,frozen_control_visual_unchanged=True,
        real_rgb_runtime_frames=count,runtime_uses_actual_600step_weights=True,trained_visual_max_change_vs_step0=delta,
        declared_motion_limits_match=True,default_v42_unchanged=True,scope='RGB reconstruction core joint training; detector/IK/final solver outside graph',browser_interaction_checked=False)
    save(RUN/'review/verification.json',evidence);print(json.dumps(evidence),flush=True)
if __name__=='__main__':main()
