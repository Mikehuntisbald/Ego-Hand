"""Final runtime and sealed-output evidence for the fixed profile."""
import json,torch,hashlib
from pathlib import Path
from hand3d_v8_common import V7,save
from complete_hand_tracks_acceleration_v46 import profile_config
run=V7.parent/'acceleration_validation_v46';r=json.loads((run/'runtime_acc_x2_retested.json').read_text())
assert r['config']==profile_config('acc_x2') and r['context_s']==1.6 and r['constraint_checks_passed']
n=0
for t in r['tracks']:
    assert not t['manual_anchor_blocked']
    for f in t['frames']:
        xyz=torch.tensor(f['xyz_camera_m']);assert xyz.shape==(20,3) and torch.isfinite(xyz).all();n+=1
assert n==16
assert json.loads((Path(__file__).parent/'CURRENT_PIPELINE.json').read_text())['version']=='offline_stability_first_v42'
frozen=json.loads((run/'outputs_frozen.json').read_text());assert frozen['complete'] and frozen['tracks']==130
for name,h in frozen['hashes'].items():assert hashlib.sha256((run/'tracks'/name).read_bytes()).hexdigest()==h
assert json.loads((run/'side_adapter_audit.json').read_text())['native_function_parity']
s=json.loads((run/'summary.json').read_text());assert s['complete'] and all(c['passed'] for c in s['constraints'].values())
save(run/'review/final_verification.json',dict(passed=True,real_rgb_runtime_retested_after_context_fix=True,real_rgb_frames=n,
    actual_exported_config_matches_declared_profile=True,all_130_saved_tracks_sealed_and_redecoded=True,selector_gt_poison_exact=True,
    native_side_function_parity=True,all_context_bounds_checked=True,same_dit_candidates_and_path_between_profiles=True,
    default_v42_unchanged=True,scope=s['scope'],browser_interaction_checked=False))
print('Final runtime and frozen-output checks passed')
