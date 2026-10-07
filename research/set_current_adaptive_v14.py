import json,shutil
from hand3d_v8_common import V7,save
RUN=V7.parent/'adaptive_projection_v14/dit_dense'
assert json.loads((RUN/'fourth_results.json').read_text())['passed']
assert json.loads((RUN/'raw_inference_checks.json').read_text())['passed']
assert json.loads((RUN/'hard_results.json').read_text())['complete']
smoke=json.loads((RUN/'dense_engineering_output.json').read_text())
frames=[f for track in smoke['tracks'] for f in track['frames']]
assert len(frames)==150 and all(f['projection_mode']=='adaptive' for f in frames)
save(RUN/'deployment_smoke_checks.json',dict(passed=True,frames=len(frames),all_adaptive=True,coordinate_frame=smoke['coordinate_frame'],unit=smoke['unit'],scope='Original RGB30FPS route and runtime seal; not additional accuracy evidence'))
pointer=V7.parent/'CURRENT_HAND_COMPLETION.json'
if pointer.exists() and not (RUN/'previous_pipeline_snapshot.json').exists():shutil.copy2(pointer,RUN/'previous_pipeline_snapshot.json')
obj=json.loads((RUN/'delivery/CURRENT_PIPELINE.json').read_text())
obj.update(goal_work_continues=True,severe_occlusion_resolved=False,next_experiment=str(V7.parent/'natural_recovery_v15'))
save(pointer,obj);print(json.dumps(obj),flush=True)
