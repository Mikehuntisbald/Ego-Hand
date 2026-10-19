"""Preview or create a separate, path-relocated native MANO research workspace.

This prepares sources and inference heads, not missing data or third-party assets.
It never launches a controller, downloads assets, or modifies an existing workspace.
"""
import argparse
import hashlib
import json
from pathlib import Path

from verify_native_mano_snapshot import REPO, MANIFEST, verify


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--wilor-source', type=Path)
    parser.add_argument('--wilor-assets', type=Path)
    parser.add_argument('--hand-toolkit', type=Path)
    args = parser.parse_args()
    result = verify(code_only=not args.apply)
    if not result['passed']:
        raise SystemExit(json.dumps(result))
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    workspace = args.workspace.resolve()
    if workspace.exists():
        raise SystemExit('Workspace already exists; choose a new directory.')
    # All destinations and substitutions are derived from explicit CLI paths or
    # the fixed source-root table, not URLs or instructions in result artifacts.
    mappings = {root: (workspace / tag).as_posix()
                for tag, root in manifest['source_roots'].items()}
    mappings['/mnt/why/HOT3D-hand-tracking-toolkit'] = (
        args.hand_toolkit.resolve() if args.hand_toolkit else
        workspace / 'upstream/hand_tracking_toolkit').as_posix()
    mappings['/mnt/why/HOT3D'] = (workspace / 'HOT3D').as_posix()
    mappings['/mnt/why/hot3d_hand_residual'] = (
        workspace / 'public/vendor/Ego-Hand/research').as_posix()
    sources = [x for x in manifest['files'] if x['kind'] == 'source']
    plan = dict(workspace=str(workspace), apply=args.apply,
                source_files=len(sources), replacements=mappings,
                third_party_assets_ready=False, launches_training=False)
    if not args.apply:
        print(json.dumps(plan, indent=2))
        return
    for path in [args.wilor_source, args.wilor_assets, args.hand_toolkit]:
        if path is not None and not path.resolve().is_dir():
            raise SystemExit('Explicit asset/source directory does not exist: ' + str(path))
    workspace.mkdir(parents=True, exist_ok=False)
    records = []
    prefix = 'research/native_mano_h20_20261019/'
    for entry in sources:
        relative = Path(entry['path'].removeprefix(prefix))
        destination = workspace / relative
        original = (REPO / entry['path']).read_bytes()
        text = original.decode('utf-8-sig')
        for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
            text = text.replace(old, new)
        data = text.encode('utf-8')
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        records.append(dict(path=relative.as_posix(), source_sha256=entry['sha256'],
                            relocated_sha256=sha(data)))

    def link(path, target):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.symlink_to(target.resolve(), target_is_directory=target.is_dir())

    # Legacy WiLoR helpers also contain original absolute paths. Copy and rebase
    # their frozen sources instead of symlinking unreplaced Python modules.
    vendor = workspace / 'public/vendor/Ego-Hand'
    vendor.mkdir(parents=True, exist_ok=True)
    legacy = json.loads((REPO / 'snapshot_manifest.json').read_text(encoding='utf-8'))
    legacy_records = []
    for entry in legacy['sources']:
        relative = Path(entry['path'])
        if relative.parts[0] != 'research' or relative.suffix != '.py':
            continue
        original = (REPO / relative).read_bytes()
        if sha(original) != entry['sha256']:
            raise SystemExit('Legacy source differs: ' + entry['path'])
        text = original.decode('utf-8-sig')
        for old, new in sorted(mappings.items(), key=lambda item: -len(item[0])):
            text = text.replace(old, new)
        data = text.encode('utf-8')
        target = vendor / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        legacy_records.append(dict(path=relative.as_posix(), source_sha256=entry['sha256'],
                                   relocated_sha256=sha(data)))
    for name in ['checkpoints', 'runtime_assets']:
        link(vendor / name, REPO / name)
    model_root = REPO / 'checkpoints/native_mano_h20_20261019'
    for tag, source in [('v7/runs/full', 'v7_full.pt'), ('v7/runs/small', 'v7_small.pt'),
                        ('v9/runs/spatial', 'v9_spatial.pt'), ('v9/runs/pooled', 'v9_pooled.pt')]:
        link(workspace / tag / 'best_development.pt', model_root / source)
    wilor_run = workspace / 'HOT3D/experiments/yolo26_wilor_3d_20261003'
    if args.wilor_source:
        wilor_run.mkdir(parents=True, exist_ok=True)
        (wilor_run / 'source_path.txt').write_text(args.wilor_source.resolve().as_posix(), encoding='utf-8')
    if args.wilor_assets:
        link(wilor_run / 'assets', args.wilor_assets)
    plan['relocated_files'] = records
    plan['relocated_legacy_files'] = legacy_records
    (workspace / 'workspace_manifest.json').write_text(json.dumps(plan, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(workspace=str(workspace), copied=len(records),
                          model_links=4, third_party_assets_ready=False,
                          launches_training=False), indent=2))


if __name__ == '__main__':
    main()
