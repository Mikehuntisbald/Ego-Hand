"""Matched actual-sampler training for predicted-side canonical RGB."""
import argparse, hashlib, sys
from pathlib import Path
from hand3d_v8_common import V7, save

CODE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['risk', 'model'], required=True)
    ap.add_argument('--variant', choices=['control', 'canonical'], required=True)
    ap.add_argument('--device', required=True)
    args = ap.parse_args()
    data = V7.parent / ('context_data_v17/control' if args.variant == 'control' else 'canonical_rgb_v19')
    run = V7.parent / 'canonical_native_v19' / args.variant
    run.mkdir(parents=True, exist_ok=True)
    if args.stage == 'risk':
        assert args.variant == 'canonical'
        import train_density_risk_v13 as worker
        worker.ROOT = data
        sys.argv = ['risk', '--arm', 'dense', '--device', args.device]
        worker.main()
        return
    source = (CODE / 'train_recovery_v15.py').read_text()
    changes = {
        'from hand3d_trajectory_data_v9 import targets': "def targets(data,extra,ids):\n    return tuple(extra[k][ids] for k in ['gt','valid','gt_uv','uv_valid'])",
        "DATA/'trajectory_targets.pt'": "DATA/'trajectory_labels.pt'",
        "base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids]": "base=data['original_base_for_evaluation'][ids];gt=data['gt'][ids]",
        "base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];b=make(data,bank,dev,ep)": "base=data['original_base_for_evaluation'][dev];b=make(data,bank,dev,ep)",
    }
    generated = source
    for before, after in changes.items():
        assert generated.count(before) == 1
        generated = generated.replace(before, after)
    snapshot = run / 'trainer_snapshot.py'
    snapshot.write_text(generated)
    save(run / 'source_provenance.json', dict(
        original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),
        replacements=changes, variant=args.variant,
        scope='Same900updates/seed/initializer/3Dlabels/XYZ/slots/policy. Canonical changes predicted-left RGB/physicalpatch positions and refits risk. No new evaluation data.'))
    ns = dict(__name__='canonical_model_v19_worker', __file__=str(snapshot))
    exec(compile(generated, str(snapshot), 'exec'), ns)
    ns['DATA'] = data
    ns['RUN'] = run
    ns['INITIAL'] = V7.parent / 'side_native_v16/consensus/uniform_adaptive/best.pt'
    sys.argv = ['train', '--arm', 'uniform_adaptive', '--device', args.device, '--steps', '900']
    ns['main']()

if __name__ == '__main__':
    main()
