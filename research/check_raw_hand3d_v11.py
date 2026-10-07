import json,subprocess,sys
from pathlib import Path
import torch
from hand3d_v8_common import V7,save
from native_projection_policy_v11 import apply
RUN=V7.parent/'native_projection_v11'

def main():
    root=Path(__file__).resolve().parent;source=V7.parent/'offline_hand3d_v7_bounded/example_input.json'
    def infer(src,dest):subprocess.run([sys.executable,str(root/'infer_hand3d_v11.py'),'--input',str(src),'--output',str(dest),'--device','cuda:2'],check=True)
    infer(source,RUN/'example_output.json');obj=json.loads(source.read_text());obj['gt']=[999.]*20
    for track in obj['tracks']:
        track['gt_hand_shape']=[-999.]*10
        for frame in track['frames']:frame.update(gt_xyz=[[999.,999.,999.]]*20,gt_side='invented',visibility=[0.]*20,gt_shape=[999.]*10)
    poison=RUN/'poison_input.json';poison.write_text(json.dumps(obj));infer(poison,RUN/'poison_output.json')
    a=json.loads((RUN/'example_output.json').read_text());b=json.loads((RUN/'poison_output.json').read_text());assert a==b
    missing=json.loads(source.read_text());frame=missing['tracks'][0]['frames'][0];frame['available_3d']=[True]*20;frame['available_3d'][1]=False;frame['confirmed_3d'][1]=False;frame['xyz_camera_m'][1]=[None,None,None]
    missing_path=RUN/'missing_input.json';missing_path.write_text(json.dumps(missing));infer(missing_path,RUN/'missing_output.json');result=json.loads((RUN/'missing_output.json').read_text());out_frame=result['tracks'][0]['frames'][0]
    assert out_frame['missing_3d_input_review_only'] and not out_frame['automatic_projection_applied'];assert out_frame['joints'][1]['xyz_camera_m'] is None
    assert torch.isfinite(torch.tensor(out_frame['joints'][1]['candidate_xyz_camera_m'])).all()
    for j in range(20):
        if j!=1:assert out_frame['joints'][j]['xyz_camera_m']==frame['xyz_camera_m'][j]
    torch.manual_seed(202610113);base=torch.randn(128,20,3)*.5;raw=base+torch.randn_like(base)*20;confirmed=torch.rand(128,20)<.15
    pred=apply(raw,base,a['policy'],confirmed);assert torch.equal(pred[confirmed],base[confirmed]);dc=(pred-base).norm(dim=-1).max();dr=((pred-pred[:,5:6])-(base-base[:,5:6])).norm(dim=-1).max();assert dc<.009501 and dr<.009501
    save(RUN/'raw_inference_checks.json',dict(passed=True,GT_field_exact_invariance=True,confirmed_checked=a['confirmed_checked'],missing_input_review_guard_passed=True,extreme_proposal_confirmed_exact=True,extreme_max_camera_mm=float(dc*1000),extreme_max_relative_mm=float(dr*1000),raw_max_camera_mm=a['max_camera_displacement_mm'],raw_max_relative_mm=a['max_relative_displacement_mm'],scope='Raw RGB engineering fixture, not human GT;128extreme proposal/lock cases; missing-input guard is not an accuracy claim'))
    print((RUN/'raw_inference_checks.json').read_text(),flush=True)

if __name__=='__main__':main()
