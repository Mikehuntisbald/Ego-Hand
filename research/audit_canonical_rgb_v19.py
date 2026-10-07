"""Independent physical correspondence checks on prepared RGB conditions."""
import json
import numpy as np
import torch
from hand3d_v8_common import V7, save
from hand3d_data_v7 import batch
from offline_rgb_encoder import crop_roi
import spatial_rgb_common as s

def main():
    torch.set_num_threads(4)
    root = V7.parent / 'canonical_rgb_v19'
    old = torch.load(V7.parent / 'context_data_v17/control/dense_data.pt', weights_only=False, mmap=True)
    new = torch.load(root / 'dense_data.pt', weights_only=False, mmap=True)
    plan = json.loads((root / 'prediction_only_encoding_plan.json').read_text())
    left = torch.tensor(plan['left_ids'])
    right = torch.tensor(plan['predicted_right'], dtype=torch.bool)
    unchanged = ['feature_ids', 'dt', 'xyz_camera_bank', 'world', 'rotation', 'translation', 'gt', 'valid', 'original_base_for_evaluation']
    exact = []
    for key in unchanged:
        if key in new:
            assert torch.equal(new[key], old[key]), key
            exact.append(key)
    for key in ['rgb_bank', 'risk_rgb_bank', 'positions_bank', 'rays_world']:
        assert torch.equal(new[key][right], old[key][right]), key
        assert torch.isfinite(new[key]).all(), key
    # Independent derivation: left canonical x maps back to 255-x before
    # undoing the 90-degree rotation. It is not a reversal of 12 feature cells.
    yy, xx = np.mgrid[:16, :12]
    canonical = np.stack([xx * 16 + 37.5, yy * 16 + 5.5], -1).reshape(192, 2)
    input_pixels = np.stack([255 - canonical[:, 1], 255 - canonical[:, 0]], -1)
    dense = json.loads((V7.parent / 'dense_sampling_v13/fresh_rows.json').read_text())
    sample = [int(i) for i in left.tolist() if int(i) <= len(dense)][::max(1, len(left)//12)][:12]
    position_delta = ray_delta = 0.
    checked = []
    for fid in sample:
        record = dense[fid - 1]
        roi = crop_roi(record['box'])
        image = __import__('cv2').imread(record['image'])
        _, rotation, focal, *_ = s.observed_crop(image, roi, record['camera'])
        eye_rays = np.c_[(input_pixels - 127.5) / focal, np.ones(192)] @ rotation.T
        pixels = s.common.from_json(record['camera']).eye_to_window(eye_rays) / 1408
        unit_eye = eye_rays / np.linalg.norm(eye_rays, axis=-1, keepdims=True)
        unit_world = unit_eye @ new['rotation'][fid].numpy().T
        pdelta = float(np.abs(pixels - new['positions_bank'][fid].numpy()).max())
        rdelta = float(np.abs(unit_world - new['rays_world'][fid].numpy()).max())
        assert pdelta < 2e-6 and rdelta < 2e-6, (fid, pdelta, rdelta)
        position_delta = max(position_delta, pdelta)
        ray_delta = max(ray_delta, rdelta)
        checked.append(fid)
    # Observation-only batch whitelist must ignore any poisoned training GT.
    ids = torch.arange(4)
    clean = batch(new, ids)
    poisoned = dict(new, gt=new['gt'] + 1234, valid=~new['valid'])
    other = batch(poisoned, ids)
    for key, value in clean.items():
        if torch.is_tensor(value):
            assert torch.equal(value, other[key]), key
    save(root / 'geometry_preflight.json', dict(passed=True, exact_inputs=exact,
        right_conditions_byte_exact=True, checked_left_observations=checked,
        max_normalized_position_delta=position_delta, max_world_ray_delta=ray_delta,
        inference_batch_gt_poison_exact=True, output_scope='Condition geometry only; not accuracy evidence'))
    print((root / 'geometry_preflight.json').read_text())

if __name__ == '__main__':
    main()
