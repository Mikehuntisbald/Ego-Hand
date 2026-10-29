"""Route measured/predicted image evidence into one fixed-camera MANO graph."""
import numpy as np
import re
from .errors import RuntimeFailure
from .schema import ns, matrix, reject_privileged, rotation, transform
from .clock import motion_pairs
from .geometry import matrix_rotvec, rotvec_matrix, slerp_matrix


def validate_observations(rows, episode):
    seen = set()
    for row in rows:
        reject_privileged(row)
        row['timestamp_ns'] = ns(row['timestamp_ns'])
        row['source_timestamp_ns'] = ns(row.get('source_timestamp_ns', row['timestamp_ns']))
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',str(row['track_id'])):
            raise RuntimeFailure('track_id','Track ID must be a safe group component')
        if row['camera_id'] not in episode.cameras:
            raise RuntimeFailure('route_camera', 'Observation camera does not exist')
        key = (row['track_id'], row['timestamp_ns'], row['camera_id'])
        if key in seen:
            raise RuntimeFailure('route_duplicate', 'Duplicate track/time/camera observation')
        seen.add(key)
        if row['right'] not in [True, False, 0, 1]:
            raise RuntimeFailure('chirality', 'Actual model chirality is required')
        for name, shape in [('state_camera51', (51,)), ('betas', (10,)), ('uv', (21, 2))]:
            matrix(row[name], shape, name)
        if row.get('observation_kind', 'wilor_self_projection') not in {'wilor_self_projection', 'independent_rgb'}:
            raise RuntimeFailure('observation_kind', 'Unrecognized image evidence source')
        if row.get('observation_kind') == 'independent_rgb' and not row.get('independent_model_sha256'):
            raise RuntimeFailure('observation_provenance', 'Independent RGB evidence must identify the executed image head')
        if len(row.get('point_valid', [True]*21)) != 21:
            raise RuntimeFailure('observation_mask', 'Need 21 point-valid flags')
        if row.get('depth_m') is not None and not .01<float(row['depth_m'])<10:
            raise RuntimeFailure('depth', 'Invalid wrist depth anchor')
    return rows


def head_at(head, timestamp_ns, max_gap_ns=100000000):
    stamps = np.array([ns(r['timestamp_ns']) for r in head], np.int64)
    if len(stamps)==0 or (np.diff(stamps)<=0).any():
        raise RuntimeFailure('head_clock', 'Head trajectory must have a unique increasing clock')
    i = int(np.searchsorted(stamps, timestamp_ns))
    if i<len(head) and stamps[i]==timestamp_ns:
        row = head[i]
        return (transform(row['world_from_device']), int(row.get('segment', 0))) if row['valid'] else (None, None)
    if not 0<i<len(head):
        return None, None
    a, b = head[i-1], head[i]
    if not a['valid'] or not b['valid'] or a.get('segment', 0)!=b.get('segment', 0) or int(stamps[i]-stamps[i-1])>max_gap_ns:
        return None, None
    w = int(timestamp_ns-stamps[i-1])/int(stamps[i]-stamps[i-1])
    A, B = transform(a['world_from_device']), transform(b['world_from_device'])
    T = np.eye(4)
    T[:3, :3] = slerp_matrix(A[:3, :3], B[:3, :3], w)
    T[:3, 3] = A[:3, 3]*(1-w)+B[:3, 3]*w
    return T, int(a.get('segment', 0))


def route(rows, head, episode, max_gap_ns):
    rows = validate_observations(rows, episode)
    seeds, mapping, routed, rejects, track_identity = [], {}, [], [], {}
    primary = episode.camera_for_role('rgb').id
    # Primary prediction seeds have priority over secondary observations. Owner,
    # chirality and timestamps are preserved; observations are not final poses.
    ordered = sorted(rows, key=lambda r: (r['timestamp_ns'], str(r['track_id']), r['camera_id']!=primary))
    for row in ordered:
        if abs(row['source_timestamp_ns']-row['timestamp_ns'])>5000000:
            rejects.append(dict(track_id=row['track_id'], timestamp_ns=row['timestamp_ns'], reason='unsynchronized_source'))
            continue
        identity = (bool(row['right']),row.get('owner','unknown'))
        if row['track_id'] in track_identity and track_identity[row['track_id']]!=identity:
            rejects.append(dict(track_id=row['track_id'],timestamp_ns=row['timestamp_ns'],reason='track_identity_change'))
            continue
        pose, segment = head_at(head, row['source_timestamp_ns'], max_gap_ns)
        if pose is None or not row.get('valid', True):
            rejects.append(dict(track_id=row['track_id'], timestamp_ns=row['timestamp_ns'], reason='invalid_head_or_observation'))
            continue
        track_identity[row['track_id']] = identity
        camera = episode.camera_at(row['camera_id'], row['timestamp_ns'])
        world_from_camera = pose@camera.T_device_from_camera
        key = (row['track_id'], row['timestamp_ns'])
        if key not in mapping:
            state = np.asarray(row['state_camera51'], np.float32).copy()
            if row.get('depth_valid') and row.get('depth_m') is not None:
                if state[2] <= .01:
                    raise RuntimeFailure('depth_ray', 'Depth initialization requires a positive camera Z')
                state[:3] *= float(row['depth_m'])/state[2]
            state[:3] = world_from_camera[:3, :3]@state[:3]+world_from_camera[:3, 3]
            state[3:6] = matrix_rotvec(world_from_camera[:3, :3]@rotvec_matrix(state[3:6]))
            mapping[key] = len(seeds)
            seeds.append(dict(track_id=row['track_id'], timestamp_ns=row['timestamp_ns'], segment=segment,
                              right=bool(row['right']), owner=row.get('owner', 'unknown'), confidence=float(row.get('confidence', 0)),
                              state=state, betas=np.asarray(row['betas'], np.float32)))
        index = mapping[key]
        if seeds[index]['right'] != bool(row['right']) or seeds[index]['owner'] != row.get('owner', 'unknown'):
            rejects.append(dict(track_id=row['track_id'], timestamp_ns=row['timestamp_ns'], reason='owner_or_chirality_conflict'))
            continue
        routed.append(dict(row, state_index=index, world_from_camera=world_from_camera,
                           camera=camera, point_valid=np.asarray(row.get('point_valid', [True]*21), bool)))
    if not seeds or not routed:
        raise RuntimeFailure('no_observations', 'No valid fixed-camera native observations remain', rejects=rejects)
    return seeds, routed, rejects
