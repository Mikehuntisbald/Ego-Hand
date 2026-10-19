"""Public trained v48 DiT on GT-free generic inputs; explicit UmeTrack proposal."""
import argparse, json, sys, hashlib, time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import os
ROOT=Path(os.environ.get('PUBLIC_HEADHAND_ROOT','/mnt/why/public_headhand_h20_20261019'))
sys.path.insert(0,str(ROOT/'vendor/Ego-Hand/research'))
from geometry import load_frames
MAPPING=[4,8,12,16,20,0,2,3,5,6,7,9,10,11,13,14,15,17,18,19]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suffix',default='');parser.add_argument('--head',default='head_clip2102');args=parser.parse_args()
    output=ROOT/f'outputs/dit_clip2102{args.suffix}';output.mkdir(exist_ok=False)
    frames=load_frames(ROOT/'data/clip-002102')
    rows=json.loads((ROOT/'outputs/wilor_clip2102/observations.json').read_text())['frames']
    head=json.loads((ROOT/f'outputs/{args.head}/headpose.json').read_text())['frames']
    source=dict(image_size=[1408,1408],tracks=[],GT_used=False)
    for tid in sorted({r['track_id'] for r in rows}):
        track=dict(id=tid,frames=[])
        for row in rows:
            if row['track_id']!=tid:continue
            fi=row['frame_index'];T=np.array(head[fi]['T_world_from_rgb']);q=Rotation.from_matrix(T[:3,:3]).as_quat()
            pose=dict(quaternion_wxyz=[q[3],*q[:3]],translation_xyz=T[:3,3].tolist())
            # Coarse observations store coordinates in NPZ, avoiding ambiguous key guesses.
            track['frames'].append(dict(image=row['image'],timestamp_s=row['timestamp_ns']/1e9,
                box_xyxy=row['box_xyxy'],box_confidence=row['confidence'],predicted_right=row['right'],
                xyz_camera_m=[],available_3d=[True]*20,confirmed_3d=[False]*20,
                camera=dict(calibration=frames[fi]['cameras']['214-1']['calibration'],T_world_from_camera=pose),
                frame_index=fi))
        source['tracks'].append(track)
    arrays=np.load(ROOT/'outputs/wilor_clip2102/observations.npz')
    lookup={(r['track_id'],r['frame_index']):i for i,r in enumerate(rows)}
    for track in source['tracks']:
        for f in track['frames']:f['xyz_camera_m']=arrays['xyz_camera'][lookup[(track['id'],f['frame_index'])],MAPPING].tolist()
    (output/'input.json').write_text(json.dumps(source))
    import complete_hand_tracks_online_v47 as online
    ck=Path('/mnt/why/HOT3D/experiments/online_rgb_iterative_v48/protected/best.pt')
    assert hashlib.sha256(ck.read_bytes()).hexdigest()=='86fd6b0a44425e44bf6a9c411d1c295043968cf2a70cc1520fb35650b49100dd'
    adapter=output/'checkpoint_adapter/dit_joint';adapter.mkdir(parents=True)
    (adapter/'best.pt').symlink_to(ck)
    online.RUN=adapter.parent
    start=time.time();result=online.complete(source,device='cuda:0',prepared=True,mode='joint',profile='acc_x2')
    result.update(public_adaptation=True,GT_used=False,seconds=time.time()-start,head_source=args.head,
                  output_semantics='UmeTrack20 candidate; requires explicit fitting to MANO21; not MANO51 checkpoint')
    path=output/'proposals20.json';path.write_text(json.dumps(result,indent=2))
    (output/'seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),GT_used=False)))
    print(json.dumps(dict(complete=True,seconds=result['seconds'],tracks=len(result['tracks']))),flush=True)

if __name__=='__main__':main()
