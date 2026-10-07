import json,shutil
from hand3d_v8_common import V7,save

run=V7.parent/'native_projection_v11';root=V7.parent
assert json.loads((run/'fresh_results.json').read_text())['passed']
assert json.loads((run/'raw_inference_checks.json').read_text())['passed']
pointer=root/'CURRENT_HAND_COMPLETION.json'
if pointer.exists() and not (run/'previous_pipeline_snapshot.json').exists():shutil.copy2(pointer,run/'previous_pipeline_snapshot.json')
obj=json.loads((run/'delivery/CURRENT_PIPELINE.json').read_text());obj.update(raw_adapter='/mnt/why/hot3d_hand_residual/infer_hand3d_v11.py',next_experiment=str(root/'dense_sampling_v13'),goal_work_continues=True,severe_occlusion_resolved=False)
save(pointer,obj);print(json.dumps(obj),flush=True)
