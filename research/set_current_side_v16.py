import json,shutil
from hand3d_v8_common import V7,save
RUN=V7.parent/'side_native_v16'
assert json.loads((RUN/'fifth_results.json').read_text())['passed'];assert json.loads((RUN/'raw_inference_checks.json').read_text())['passed'];assert json.loads((RUN/'hard_results.json').read_text())['complete'];assert (RUN/'deployment_seal.json').exists()
pointer=V7.parent/'CURRENT_HAND_COMPLETION.json'
if pointer.exists() and not (RUN/'previous_pipeline_snapshot.json').exists():shutil.copy2(pointer,RUN/'previous_pipeline_snapshot.json')
obj=json.loads((RUN/'delivery/CURRENT_PIPELINE.json').read_text());obj.update(goal_work_continues=True,next_experiment=str(V7.parent/'context_merge_v17'));save(pointer,obj);print(json.dumps(obj),flush=True)
