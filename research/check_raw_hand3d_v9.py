import json,subprocess,sys
from pathlib import Path
from hand3d_v8_common import V7,save
RUN=V7.parent/'proposal_critic_v9'
def main():
    root=Path(__file__).resolve().parent;source=V7.parent/'offline_hand3d_v7_bounded/example_input.json';p=json.loads(source.read_text());p['gt']=[999.]*20
    for track in p['tracks']:
        track['gt_hand_shape']=[-999.]*10
        for frame in track['frames']:frame.update(gt_xyz=[[999.,999.,999.]]*20,gt_side='invented',visibility=[0.]*20,gt_shape=[999.]*10)
    dest=RUN/'poison_input.json';dest.write_text(json.dumps(p));subprocess.run([sys.executable,str(root/'infer_hand3d_v9.py'),'--input',str(dest),'--output',str(RUN/'poison_output.json'),'--device','cuda:2'],check=True)
    original=json.loads((RUN/'example_output.json').read_text());poison=json.loads((RUN/'poison_output.json').read_text());assert original==poison
    save(RUN/'raw_inference_checks.json',dict(passed=True,GT_field_exact_invariance=True,confirmed_checked=original['confirmed_checked'],scope='Raw RGB engineering fixture; locked points are not human GT'))
    print((RUN/'raw_inference_checks.json').read_text(),flush=True)
if __name__=='__main__':main()
