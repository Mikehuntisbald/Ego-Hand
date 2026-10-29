"""Internal native output and Aria research exporter, both on the full source clock."""
from pathlib import Path
import numpy as np
from .clock import interpolate_native
from .observations import head_at
from .io import atomic_json
from .errors import RuntimeFailure


def write(episode, seeds, result, head, codec, directory, config, invalid_times=None,active_range=None):
    import h5py
    directory = Path(directory)
    clock = episode.source_clock_ns
    tracks = {}
    for track in dict.fromkeys(r['track_id'] for r in seeds):
        ids = np.array([i for i, r in enumerate(seeds) if r['track_id']==track])
        order = np.argsort([seeds[i]['timestamp_ns'] for i in ids])
        ids = ids[order]
        rows = interpolate_native([seeds[i]['timestamp_ns'] for i in ids], result['state'][ids],
                                   result['betas'][ids], result['right'][ids], [True]*len(ids),
                                   [seeds[i]['segment'] for i in ids], clock, codec,
                                   config['delivery']['max_interpolation_gap_ns'])
        tracks[track] = rows
    frames, pose_rows, pose_valid = [], [], []
    for index, stamp in enumerate(clock):
        pose, segment = head_at(head, int(stamp), config['headpose']['max_gap_ns'])
        if active_range is not None and not active_range[0]<=int(stamp)<=active_range[1]:
            pose,segment=None,None
        good = pose is not None
        pose_rows.append(pose if good else np.full((4, 4), np.nan))
        pose_valid.append(good)
        hands = []
        for track, rows in tracks.items():
            row = rows[index]
            blocked = int(stamp) in (invalid_times or {}).get(str(track),set())
            valid = row['valid'] and good and row['segment']==segment and not blocked
            item = dict(track_id=track, valid=bool(valid), interpolated=bool(row['interpolated']), owner='unknown')
            if valid:
                native = codec.export(row['state'][None], row['betas'][None], np.array([row['right']]))
                item.update(right=row['right'], state_world51=row['state'], joints_world_m=native['joints'][0],
                            global_orient=native['global_orient'][0], hand_pose=native['hand_pose'][0],
                            betas=native['betas'][0], transl=native['transl'][0], reflection=native['reflection'][0])
            hands.append(item)
        frames.append(dict(source_index=episode.frames[index]['source_index'], timestamp_ns=int(stamp),
                           head_valid=good, camera_segment=segment, world_from_device=pose,
                           hands=hands))
    if not any(h['valid'] for frame in frames for h in frame['hands']):
        raise RuntimeFailure('empty_delivery', 'No valid hand state can be delivered on the source clock')
    output = dict(schema='ego-hand-native-1', units='m', skeleton='MANO21/OpenPose/wrist0',
                  source_id=episode.source_id, episode_id=episode.episode_id,
                  exporter='aria-research-mano21' if config['delivery']['format']=='aria' else 'internal',
                  mego_formal_export=False, timestamps='original integer ns; no invented 60Hz clock',
                  geometry='right-canonical MANO plus explicit reflection; transl is not wrist XYZ',
                  head_pose_frame='device; physical camera extrinsics are retained in input_contract.json',
                  world_gauge='stereo VO rig gauge, without GT alignment',
                  headpose='stereo VO without IMU; fixed in hand BA', GT_used=False, frames=frames)
    atomic_json(directory / 'joints.json', output)
    with h5py.File(directory / 'bundle.h5.partial', 'w') as h5:
        h5.attrs['schema'] = 'ego-hand-native-1'
        h5.attrs['mego_formal_export'] = False
        h5.attrs['head_pose_frame']='device; see input_contract.json for physical camera extrinsics'
        h5.create_dataset('timestamp_ns', data=clock, dtype='i8')
        h5.create_dataset('source_index', data=[f['source_index'] for f in episode.frames], dtype='i8')
        camera = h5.create_group('head_camera')
        camera.create_dataset('timestamp_ns', data=clock, dtype='i8')
        camera.create_dataset('validmask', data=np.asarray(pose_valid, bool))
        camera.create_dataset('world_from_device', data=np.asarray(pose_rows, np.float32))
        for track, rows in tracks.items():
            group = h5.create_group('tracks/'+str(track))
            group.create_dataset('timestamp_ns', data=clock, dtype='i8')
            valid = np.array([f['hands'][list(tracks).index(track)]['valid'] for f in frames])
            group.create_dataset('validmask', data=valid)
            group.create_dataset('interpolated', data=np.array([r['interpolated'] for r in rows]))
            for name, shape in [('joints_world_m', (21, 3)), ('state_world51', (51,)), ('global_orient', (1, 3, 3)),
                                ('hand_pose', (15, 3, 3)), ('betas', (10,)), ('transl', (3,)), ('reflection', (3,))]:
                values = np.full((len(clock), *shape), np.nan, np.float32)
                for i, frame in enumerate(frames):
                    item = frame['hands'][list(tracks).index(track)]
                    if item['valid']:
                        values[i] = item[name]
                group.create_dataset(name, data=values)
    (directory / 'bundle.h5.partial').replace(directory / 'bundle.h5')
    return output


def verify(directory):
    import h5py
    from .io import read_json
    root = Path(directory)
    data = read_json(root / 'joints.json')
    clock = np.array([f['timestamp_ns'] for f in data['frames']], np.int64)
    with h5py.File(root / 'bundle.h5', 'r') as h5:
        if not np.array_equal(h5['timestamp_ns'][:], clock) or not np.array_equal(h5['head_camera/timestamp_ns'][:], clock):
            raise RuntimeFailure('delivery_clock', 'Head and delivery source clocks differ')
        if not np.array_equal(h5['source_index'][:],[f['source_index'] for f in data['frames']]):
            raise RuntimeFailure('delivery_index','JSON/H5 source indices differ')
        for i,frame in enumerate(data['frames']):
            if bool(h5['head_camera/validmask'][i])!=frame['head_valid']:
                raise RuntimeFailure('delivery_validity','JSON/H5 head validity differs')
            if frame['head_valid'] and not np.array_equal(h5['head_camera/world_from_device'][i],np.asarray(frame['world_from_device'],np.float32)):
                raise RuntimeFailure('delivery_geometry','JSON/H5 head pose differs')
        valid_count = 0
        for name, group in h5['tracks'].items():
            if not np.array_equal(group['timestamp_ns'][:], clock):
                raise RuntimeFailure('delivery_clock', 'Track clock differs')
            for i, f in enumerate(data['frames']):
                item = next(r for r in f['hands'] if str(r['track_id'])==name)
                if bool(group['validmask'][i]) != item['valid']:
                    raise RuntimeFailure('delivery_validity', 'JSON/H5 valid masks differ')
                if item['valid']:
                    valid_count += 1
                    if not np.array_equal(group['joints_world_m'][i], np.asarray(item['joints_world_m'], np.float32)):
                        raise RuntimeFailure('delivery_geometry', 'JSON/H5 joints differ')
        return dict(passed=True, frames=len(clock), valid_hand_frames=valid_count, source_clock_preserved=True)
