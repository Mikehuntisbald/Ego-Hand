"""Sampling and interpolation preserve the input integer source clock."""
import numpy as np
from .geometry import slerp_matrix, rotvec_matrix, matrix_rotvec
from .errors import RuntimeFailure


def sample_indices(timestamps, fps=None):
    t = np.asarray(timestamps, np.int64)
    if len(t) == 0 or (np.diff(t)<=0).any():
        raise RuntimeFailure('clock', 'Sampling requires a nonempty increasing ns clock')
    if fps is None:
        return np.arange(len(t), dtype=int)
    if not np.isfinite(fps) or fps <= 0:
        raise RuntimeFailure('fps', 'Sampling FPS must be positive')
    # Integer differences are computed before conversion; large epoch values
    # never pass through float64, which would erase nanosecond distinctions.
    relative = t-t[0]
    targets = np.arange(0, int(relative[-1])+1, 1e9/fps)
    right = np.searchsorted(relative, targets).clip(0, len(t)-1)
    left = (right-1).clip(0)
    indices = np.where(np.abs(relative[left]-targets)<=np.abs(relative[right]-targets), left, right)
    return np.unique(np.r_[indices, 0, len(t)-1]).astype(int)


def motion_pairs(timestamps, tracks, segments, valid, max_gap_ns):
    t = np.asarray(timestamps, np.int64)
    pairs, triples = [], []
    tracks, segments, valid = np.asarray(tracks), np.asarray(segments), np.asarray(valid, bool)
    for track in dict.fromkeys(tracks.tolist()):
        ids = np.flatnonzero(tracks==track)
        good_pairs = set()
        for a, b in zip(ids[:-1], ids[1:]):
            if valid[a] and valid[b] and segments[a]==segments[b] and 0<int(t[b]-t[a])<max_gap_ns:
                pairs.append((a, b))
                good_pairs.add((a, b))
        for a, b, c in zip(ids[:-2], ids[1:-1], ids[2:]):
            if (a, b) in good_pairs and (b, c) in good_pairs:
                triples.append((a, b, c))
    return np.asarray(pairs, int).reshape(-1, 2), np.asarray(triples, int).reshape(-1, 3)


def interpolate_native(timestamps, states, betas, chirality, valid, segments,
                       source_clock, decode, max_gap_ns=100000000):
    """Interpolate wrist, SO(3) and beta, then FK; never interpolate XYZ joints."""
    t = np.asarray(timestamps, np.int64)
    if len(t) and (np.diff(t)<=0).any():
        raise RuntimeFailure('track_clock', 'One track must have unique increasing timestamps')
    states, betas = np.asarray(states), np.asarray(betas)
    answer = []
    for stamp in np.asarray(source_clock, np.int64):
        i = int(np.searchsorted(t, stamp))
        row = dict(timestamp_ns=int(stamp), valid=False, interpolated=False)
        if i<len(t) and t[i]==stamp:
            if valid[i]:
                row.update(valid=True, state=states[i].copy(), betas=betas[i].copy(), right=bool(chirality[i]), segment=int(segments[i]))
        elif 0<i<len(t) and valid[i-1] and valid[i] and segments[i-1]==segments[i] and chirality[i-1]==chirality[i] and int(t[i]-t[i-1])<=max_gap_ns:
            w = int(stamp-t[i-1])/int(t[i]-t[i-1])
            state = states[i-1].copy()
            state[:3] = states[i-1, :3]*(1-w)+states[i, :3]*w
            for start, end in [(3, 6), (6, 51)]:
                a = rotvec_matrix(states[i-1, start:end].reshape(-1, 3))
                b = rotvec_matrix(states[i, start:end].reshape(-1, 3))
                state[start:end] = matrix_rotvec(slerp_matrix(a, b, w)).reshape(-1)
            row.update(valid=True, state=state, betas=betas[i-1]*(1-w)+betas[i]*w,
                       right=bool(chirality[i]), segment=int(segments[i]), interpolated=True)
        if row['valid']:
            row['state'] = np.asarray(row['state'], np.float32)
            row['betas'] = np.asarray(row['betas'], np.float32)
            row['joints'] = np.asarray(decode(row['state'][None], row['betas'][None],
                                            np.array([row['right']]))[0], np.float32)
        answer.append(row)
    return answer
