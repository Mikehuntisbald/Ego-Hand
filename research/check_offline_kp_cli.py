import copy,json
import torch
from pathlib import Path
from infer_offline_kp import complete
from offline_kp_model import KeypointCompleter
RUN=Path('/mnt/why/HOT3D/experiments/offline_keypoint_diffusion_v1')
torch.set_num_threads(4)
inp=json.loads((RUN/'delivery/example_input.json').read_text())
out=json.loads((RUN/'delivery/example_output.json').read_text())
checked=0
for t,u in zip(inp['tracks'],out['tracks']):
    for f,g in zip(t['frames'],u['frames']):
        for j,seen in enumerate(f['observed']):
            if seen:assert f['xy_px'][j]==g['points'][j]['xy_px'];checked+=1
            assert not g['points'][j]['reviewed'] and g['points'][j]['review_required']
empty=copy.deepcopy(inp['tracks'][0])
for f in empty['frames']:f['observed']=[False]*20;f['xy_px']=[None]*20
ck=torch.load(RUN/'sealed/dit.pt',map_location='cpu',weights_only=False)
model=KeypointCompleter('dit').eval();model.load_state_dict(ck['model'])
abstained=complete(empty,model,None,'cpu')
assert all(p['xy_px'] is None and not p['available'] for f in abstained['frames'] for p in f['points'])
checks=dict(gpu_json_cli_executed=True,cpu_json_cli_executed=(RUN/'example_cpu_output.json').exists(),
    known_input_coordinates_preserved=checked,no_context_returns_null=True,all_automatic_points_require_review=True,
    review_javascript_syntax_passed=True,review_browser_interaction_tested=False,
    browser_limitation='Local file navigation blocked by browser security policy; no workaround attempted')
(RUN/'delivery/cli_checks.json').write_text(json.dumps(checks,indent=2))
(RUN/'status.json').write_text(json.dumps(dict(stage='first_version_delivered',training_complete=True,controlled_gap_evaluation_complete=True,
    single_finger_improvement_vs_linear=True,dit_better_than_regression=False,whole_hand_gap_passed=False,natural_occlusion_validated=False,
    annotation_ready_without_human_review=False),indent=2))
print(json.dumps(checks,indent=2))
