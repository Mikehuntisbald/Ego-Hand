"""Build a separate Linux research workspace; preview by default, never overwrite."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Apply the previewed plan in a separate workspace')
    parser.add_argument('--wilor-source', type=Path)
    parser.add_argument('--wilor-assets', type=Path)
    parser.add_argument('--hand-toolkit', type=Path)
    parser.add_argument('--sam2-source', type=Path)
    args = parser.parse_args()
    root = args.workspace.expanduser().resolve()
    if args.apply and root == REPO:
        raise ValueError('Workspace must be separate from the repository')
    if any(char in root.as_posix() for char in "'\"\\"):
        raise ValueError('Use a Linux workspace path without quote or backslash characters')
    manifest = json.loads((REPO/'snapshot_manifest.json').read_text())
    replacements = [
        ('/mnt/why/HOT3D-hand-tracking-toolkit', (root/'third_party/hand_tracking_toolkit').as_posix()),
        ('/mnt/why/hot3d_hand_residual', (root/'code').as_posix()),
        ('/mnt/why/HOT3D', (root/'HOT3D').as_posix()),
    ]
    writes = []
    links = []
    for entry in manifest['sources']:
        source = REPO/entry['path']
        text = source.read_text(encoding='utf-8-sig')
        for old, new in replacements:
            text = text.replace(old, new)
        ast.parse(text, filename=source.name)
        writes.append((root/'code'/source.name, text.encode('utf-8')))
    for entry in manifest['checkpoints']:
        links.append((root/'HOT3D'/entry['workspace_relative_path'], REPO/entry['path']))
    writes.append((root/'HOT3D/experiments/side_data_v16/consensus/risk_dense/calibration.json',
                   (REPO/'runtime_assets/risk_calibration.json').read_bytes()))
    if args.wilor_source:
        source = args.wilor_source.expanduser().resolve()
        if not (source/'wilor').is_dir():
            raise ValueError('--wilor-source must contain the wilor Python package')
        writes.append((root/'HOT3D/experiments/yolo26_wilor_3d_20261003/source_path.txt',
                       (source.as_posix()+'\n').encode()))
    if args.wilor_assets:
        assets = args.wilor_assets.expanduser().resolve()
        for name in ['model_config.yaml', 'MANO_RIGHT.pkl', 'mano_mean_params.npz', 'wilor_final.mirror.ckpt']:
            source = assets/name
            if not source.is_file():
                raise FileNotFoundError(source)
            links.append((root/'HOT3D/experiments/yolo26_wilor_3d_20261003/assets'/name, source))
        helper = assets/'wilor_detector.pt'
        if helper.exists():
            links.append((root/'HOT3D/experiments/detector_compare_wilor_20261003/wilor_detector.pt', helper))
    if args.hand_toolkit:
        source = args.hand_toolkit.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        links.append((root/'third_party/hand_tracking_toolkit', source))
    if args.sam2_source:
        source = args.sam2_source.expanduser().resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        links.append((root/'code/third_party_v51/sam2', source))
    plan = dict(workspace=str(root), apply=args.apply, source_files=len(manifest['sources']),
                checkpoints=len(manifest['checkpoints']), writes=len(writes), links=len(links),
                starts_training=False, starts_automation=False,
                external_assets='Provide WiLoR source/assets and toolkit before GPU inference; SAM2 is optional for its historical control')
    print(json.dumps(plan, indent=2))
    if not args.apply:
        return
    # Check the complete plan and hashes before the first write.
    for entry in manifest['checkpoints']:
        source = REPO/entry['path']
        if not source.is_file() or source.stat().st_size != entry['bytes'] or digest(source) != entry['sha256']:
            raise ValueError(f'Missing or altered LFS checkpoint: {entry["path"]}')
    for target, content in writes:
        target.relative_to(root)
        if target.is_symlink() or (target.exists() and (not target.is_file() or target.read_bytes() != content)):
            raise FileExistsError(f'Refusing to overwrite {target}')
    for target, source in links:
        target.relative_to(root)
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists() or target.is_symlink():
            if not target.is_symlink() or target.resolve() != source.resolve():
                raise FileExistsError(f'Refusing to replace {target}')
    for target, content in writes:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(content)
    for target, source in links:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    (root/'HOT3D/experiments/offline_hand3d_v7').mkdir(parents=True, exist_ok=True)
    print('Workspace prepared; no process was launched.')


if __name__ == '__main__':
    main()
