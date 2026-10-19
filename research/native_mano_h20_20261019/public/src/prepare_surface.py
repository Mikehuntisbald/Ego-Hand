"""Cache actual MANO surfaces and validate parameter-to-joint reconstruction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, '/mnt/why/public_headhand_h20_20261019/vendor/Ego-Hand/research')
import wilor_eval_common
from mano_model import load_decoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coarse', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    data = np.load(args.coarse/'observations.npz')
    decoder = load_decoder()
    with torch.inference_mode():
        joints, vertices = decoder(torch.from_numpy(data['state']).cuda(), torch.from_numpy(data['betas']).cuda(),
                                   torch.from_numpy(data['right']).cuda(), vertices=True)
    xyz, mesh = joints.cpu().numpy(), vertices.cpu().numpy()
    difference = float(np.abs(xyz-data['xyz_camera']).max())
    assert difference < 1e-6, difference
    np.savez_compressed(args.output, joints=xyz, vertices=mesh, faces=decoder.layer.faces)
    receipt = dict(complete=True, FK_max_absolute_difference_m=difference, GT_used=False,
                   sha256=hashlib.file_digest(args.output.open('rb'), 'sha256').hexdigest())
    args.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
