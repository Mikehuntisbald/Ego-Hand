"""Prediction-only candidate risk and trackwise MANO protection; never GT ranking."""
import numpy as np
from .geometry import project, rotvec_matrix, matrix_rotvec, slerp_matrix


def candidate_cost(row):
    q = row.get('quality', {})
    finite = np.isfinite(np.asarray(row['state_camera51'])).all()
    if not finite or row.get('owner_conflict') or row.get('chirality_conflict'):
        return float('inf')
    return (1-float(row.get('confidence', 0))) + float(q.get('reprojection_error_px', 0))/20 + float(q.get('crop_truncation', 0))


def select_path(candidates, timestamps, max_gap_ns=100000000):
    """Whole-track DP with future evidence; each option is an actual model run."""
    costs, back = [], []
    for t, options in enumerate(candidates):
        unary = np.array([candidate_cost(x) for x in options])
        if t==0 or timestamps[t]-timestamps[t-1]>max_gap_ns:
            costs.append(unary)
            back.append(np.full(len(options), -1))
            continue
        dt = (timestamps[t]-timestamps[t-1])/1e9
        previous = candidates[t-1]
        transition = np.empty((len(previous), len(options)))
        for a, old in enumerate(previous):
            for b, new in enumerate(options):
                conflict = old['right']!=new['right'] or old.get('owner', 'unknown')!=new.get('owner', 'unknown')
                delta = np.linalg.norm(np.asarray(old['state_camera51'])[:3]-np.asarray(new['state_camera51'])[:3])
                transition[a, b] = np.inf if conflict else .01*delta/max(dt, 1e-6)
        values = costs[-1][:, None]+transition
        parent = values.argmin(0)
        costs.append(unary+values[parent, np.arange(len(options))])
        back.append(parent)
    chosen = [0]*len(candidates)
    for t in range(len(candidates)-1, -1, -1):
        if t==len(candidates)-1 or back[t+1][chosen[t+1]]<0:
            chosen[t] = int(costs[t].argmin())
        else:
            chosen[t] = int(back[t+1][chosen[t+1]])
    return chosen


def reprojection_mean(joints, observations):
    errors = []
    for obs in observations:
        T = obs['world_from_camera']
        cam = (joints[obs['state_index']]-T[:3, 3])@T[:3, :3]
        uv = project(cam, obs['camera'])
        valid = np.asarray(obs['point_valid'], bool)
        errors.extend(np.linalg.norm(uv-np.asarray(obs['uv']), axis=-1)[valid].tolist())
    return float(np.mean(errors)) if errors else float('inf')


def protect(baseline, proposal, seeds, observations, codec, limits):
    if not np.array_equal(baseline['right'],proposal['right']):
        from .errors import RuntimeFailure
        raise RuntimeFailure('proposal_chirality','Proposal cannot change the actual model convention')
    if np.array_equal(baseline['state'], proposal['state']) and np.array_equal(baseline['betas'], proposal['betas']):
        return baseline, dict(enabled=limits['enabled'], unchanged_control=True, GT_accuracy_guarantee=False)
    if not limits['enabled']:
        return proposal, dict(enabled=False, GT_accuracy_guarantee=False)
    result = dict(proposal)
    state = baseline['state'].copy()
    beta = baseline['betas'].copy()
    reports = []
    for track in dict.fromkeys(r['track_id'] for r in seeds):
        ids = np.array([i for i, row in enumerate(seeds) if row['track_id']==track])
        ref = baseline['joints'][ids]
        relevant = [o for o in observations if o['state_index'] in set(ids.tolist())]
        old_reproj = reprojection_mean(baseline['joints'], relevant)
        a, b = baseline['state'][ids], proposal['state'][ids]
        def trial(alpha):
            s = a*(1-alpha)+b*alpha
            for start, end in [(3, 6), (6, 51)]:
                A = rotvec_matrix(a[:, start:end].reshape(-1, 3))
                B = rotvec_matrix(b[:, start:end].reshape(-1, 3))
                s[:, start:end] = matrix_rotvec(slerp_matrix(A, B, alpha)).reshape(len(ids), -1)
            be = baseline['betas'][ids]*(1-alpha)+proposal['betas'][ids]*alpha
            xyz = codec(s, be, baseline['right'][ids])
            displacement = np.linalg.norm(xyz-ref, axis=-1).reshape(-1)
            all_xyz = baseline['joints'].copy()
            all_xyz[ids] = xyz
            metrics = dict(rmse_m=float(np.sqrt(np.mean(displacement**2))), p95_m=float(np.percentile(displacement, 95)),
                           maximum_m=float(displacement.max()), reprojection_increase_px=reprojection_mean(all_xyz, relevant)-old_reproj)
            okay = all(metrics[k]<=limits[k]+1e-9 for k in metrics)
            return okay, s, be, metrics
        okay, s, be, metrics = trial(1.)
        alpha = 1.
        if not okay:
            lo, hi = 0., 1.
            for _ in range(16):
                mid = (lo+hi)/2
                if trial(mid)[0]:
                    lo = mid
                else:
                    hi = mid
            alpha = lo
            okay, s, be, metrics = trial(alpha)
        if not okay:
            alpha, s, be = 0., a, baseline['betas'][ids]
            metrics = dict(rmse_m=0.,p95_m=0.,maximum_m=0.,reprojection_increase_px=0.)
        state[ids], beta[ids] = s, be
        reports.append(dict(track_id=track, alpha=alpha, **metrics))
    result.update(state=state, betas=beta, joints=codec(state, beta, baseline['right']))
    return result, dict(enabled=True, tracks=reports, parameter_interpolation=True, GT_accuracy_guarantee=False)
