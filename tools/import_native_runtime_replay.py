"""Import sealed historical MANO predictions for geometry/runtime regression only.

This does NOT reproduce the original detector, image observations or live VO.
"""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ego_hand_runtime.io import read_json, sha256, atomic_json, seal
from ego_hand_runtime.schema import Episode
from ego_hand_runtime.geometry import project, rotvec_matrix, matrix_rotvec
from ego_hand_runtime.errors import RuntimeFailure


def verified_old_file(path, receipt):
    path, receipt = Path(path), read_json(receipt)
    if receipt.get('GT_used') is not False:
        raise RuntimeFailure('old_seal','Historical inference GT isolation missing')
    digest = receipt.get('files',{}).get(path.name,receipt.get('sha256'))
    if digest!=sha256(path):
        raise RuntimeFailure('old_seal','Historical inference bytes differ',file=str(path))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['episode','parameters','parameters-seal','head-json','head-seal','output']:
        p.add_argument('--'+name,required=True)
    args = p.parse_args()
    verified_old_file(args.parameters,args.parameters_seal);verified_old_file(args.head_json,args.head_seal)
    episode = Episode.load(args.episode)
    source_head = read_json(args.head_json)
    if source_head.get('GT_used') is not False or 'cuVSLAM' not in source_head.get('backend',''):
        raise RuntimeFailure('replay_head','Require sealed recorded public stereo VO, not a dataset reference trajectory')
    old_head = {r['timestamp_ns']:r for r in source_head['frames']}
    primary = episode.camera_for_role('rgb')
    head = []
    for stamp in episode.source_clock_ns:
        row = old_head.get(int(stamp))
        valid = bool(row and row['valid'] and not row.get('temporal_boundary_hold',False))
        value = dict(timestamp_ns=int(stamp),valid=valid,segment=row.get('segment',0) if row else -1)
        if valid:
            camera = episode.camera_at(primary.id,int(stamp))
            value['world_from_device'] = (np.array(row['T_world_from_rgb'])@np.linalg.inv(camera.T_device_from_camera)).tolist()
        head.append(value)
    with np.load(args.parameters) as z:
        rows = []
        for i,stamp in enumerate(z['timestamp_ns']):
            if int(stamp) not in old_head:
                raise RuntimeFailure('replay_clock','Native prediction timestamp has no recorded VO pose')
            T = np.asarray(old_head[int(stamp)]['T_world_from_rgb'])
            state = z['state_world51'][i].copy()
            state[:3] = (state[:3]-T[:3,3])@T[:3,:3]
            state[3:6] = matrix_rotvec(T[:3,:3].T@rotvec_matrix(state[3:6]))
            camera = episode.camera_at(primary.id,int(stamp))
            rows.append(dict(timestamp_ns=int(stamp),source_timestamp_ns=int(stamp),camera_id=primary.id,
                             track_id=int(z['track_id'][i]),right=bool(z['right'][i]),owner='unknown',confidence=0.,
                             state_camera51=state.tolist(),betas=z['betas_shared'][z['group'][i]].tolist(),
                             uv=project(z['joints_camera'][i],camera).tolist(),point_valid=[True]*21,
                             box_scale_px=100.,observation_kind='wilor_self_projection',
                             depth_valid=False,depth_m=None,
                             observation_detail='self-projection of archived BA parameters; NOT original independent RGB evidence'))
    out = Path(args.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'head.json',head);atomic_json(out/'observations.json',rows)
    atomic_json(out/'provenance.json',dict(kind='recorded_predictions',
                purpose='geometry/scheduling/source-clock regression; no new accuracy or frontend evidence',
                parameters_sha256=sha256(args.parameters),head_sha256=sha256(args.head_json),
                source_episode_sha256=sha256(episode.source_path),GT_used=False))
    seal(out,['head.json','observations.json','provenance.json'],binding=sha256(args.parameters),backend='historical_predictions')
    print(dict(output=str(out.resolve()),observations=len(rows),head_valid=sum(r['valid'] for r in head)))


if __name__=='__main__':main()
