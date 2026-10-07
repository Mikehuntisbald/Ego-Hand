import json,subprocess,sys
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import RUN,save
from bounded_policy_v8 import apply
from hand3d_temporal_v7 import WRIST

def main():
    root=Path(__file__).resolve().parent;source=RUN.parent/'offline_hand3d_v7_bounded/example_input.json';obj=json.loads(source.read_text());poison=json.loads(source.read_text());poison['gt_xyz']=[999]*20
    for track in poison['tracks']:
        track['visibility']=[1]*20
        for f in track['frames']:f.update(gt_xyz=[[99.,99.,99.]]*20,gt_handedness='impossible',visibility=[0.]*20,gt_shape=[-999.]*10)
    dest=RUN/'poison_input.json';dest.write_text(json.dumps(poison))
    subprocess.run([sys.executable,str(root/'infer_hand3d_v8.py'),'--input',str(dest),'--output',str(RUN/'poison_output.json'),'--device','cuda:0'],check=True)
    baseline=json.loads((RUN/'example_output.json').read_text());test=json.loads((RUN/'poison_output.json').read_text());assert baseline==test,'Forbidden GT fields changed inference'
    rng=torch.Generator().manual_seed(202610087);base=torch.randn(1000,20,3,generator=rng);proposal=base+torch.randn(1000,20,3,generator=rng)*10;confirmed=torch.rand(1000,20,generator=rng)<.4;confirmed[:,WRIST]=True
    output=apply(proposal,base,dict(cap_m=.0095,strength=.5),confirmed);delta=output-base;relative=delta-delta[:,WRIST:WRIST+1];assert delta.norm(dim=-1).max()<.009502 and relative.norm(dim=-1).max()<.009502;assert torch.equal(output[confirmed],base[confirmed])
    save(RUN/'raw_inference_checks.json',dict(passed=True,forbidden_gt_fields_exact_invariance=True,confirmed_points_checked=baseline['confirmed_checked'],max_camera_displacement_mm=baseline['max_camera_displacement_mm'],max_relative_displacement_mm=baseline['max_relative_displacement_mm'],extreme_proposal_and_locked_wrist_bound_checked=True,scope='Engineering fixture; locks are not human GT'))
    print((RUN/'raw_inference_checks.json').read_text(),flush=True)

if __name__=='__main__':main()
