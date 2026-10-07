"""Parameter-generator observations -> shared-shape whole-track solver.

No independent XYZ corrections are allowed after the kinematic decoder.
Temporal consistency is checked in world coordinates using real timestamps.
"""
import torch
from parameter_codec_v31 import ParameterCodec
from joint_mano_model_v29 import temporal_indices, six_to_rotation, rotation_to_six


class ParameterTrajectoryDecoder(ParameterCodec):
    def initialize(self, obs, cache, rows):
        device = cache['base'].device
        group, keys, pair, triple = temporal_indices(rows, device)
        flat = obs['parameter_state'].float().flatten(1)
        probability = ((flat[:, 9:29].clamp(-.9999, .9999) + 1) / 2)
        local = torch.logit(probability)
        rotation = cache['rotation'] @ six_to_rotation(flat[:, 3:9])
        root = torch.einsum('nj,nkj->nk', flat[:, :3]*.1, cache['rotation']) + cache['translation']
        beta = torch.stack([flat[group==g, 29:34].median(0).values for g in range(len(keys))]).clamp(-3.999, 3.999)
        # Chirality is static on one predicted track. Prediction-only majority
        # voting prevents a left/right switch inside the joint reconstruction.
        right = obs['right'].clone().long()
        for g in range(len(keys)):
            mask = group==g
            right[mask] = int(right[mask].float().mean() >= .5)
        return dict(local=local, global_six=rotation_to_six(rotation), root=root,
                    beta=beta, group=group, keys=keys, pair=pair, triple=triple,
                    sign=1-2*right.float())

    @torch.no_grad()
    def reconstruction_check(self, obs, cache, rows):
        initial = self.initialize(obs, cache, rows)
        flat = obs['parameter_state'].float().flatten(1)
        raw = torch.logit((flat[:, 9:29].clamp(-.9999, .9999)+1)/2)
        rotation = rotation_to_six(cache['rotation'] @ six_to_rotation(flat[:, 3:9]))
        n = len(rows)
        predicted = self(raw, rotation, initial['root'], flat[:, 29:34],
                         torch.arange(n, device=flat.device), 1-2*obs['right'].float())
        expected = torch.einsum('njc,nkc->njk', obs['xyz'], cache['rotation']) + cache['translation'][:, None]
        error = (predicted-expected).norm(dim=-1)*1000
        return dict(passed=bool(error.max()<.02), parity_mean_mm=float(error.mean()), parity_max_mm=float(error.max()),
                    static_shape='One training-PCA shape vector per predicted track',
                    static_side='Prediction-only track majority', subject_calibration_inference=False)

