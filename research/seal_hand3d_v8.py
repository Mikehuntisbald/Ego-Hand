import json,hashlib,time,shutil
from pathlib import Path
from hand3d_v8_common import RUN,save
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
selection=json.loads((RUN/'development_selection.json').read_text());assert selection['approved_on_development'];selected=selection['selected'];folder=selected['folder']
assert (RUN/'fresh_ready.json').exists() and not (RUN/'fresh_results.json').exists()
controls={}
for arm,trials in json.loads((RUN/'calibration_trials.json').read_text()).items():
    feasible=[r for r in trials if r['feasible'] and r['policy']['cap_m'] is not None]
    if feasible:controls[arm]=min(feasible,key=lambda r:r['metrics']['camera_mm']+.75*r['metrics']['relative_mm'])['policy']
root=Path(__file__).resolve().parent;code={p.name:sha(p) for p in root.glob('*v8.py')}
obj=dict(folder=folder,policy=selected['policy'],checkpoint_sha256=sha(RUN/folder/'best.pt'),manifest_sha256=sha(RUN/'fresh_manifest.json'),code_sha256=code,controls=controls,controls_sha256={k:sha(RUN/k/'best.pt') for k in controls},created_unix=time.time(),selection_closed=True,scope='Untouched clip metrics; reused subjects/sequences. No subject/pretraining-disjoint claim.')
save(RUN/'fresh_evaluation_seal.json',obj);print(json.dumps(obj),flush=True)
