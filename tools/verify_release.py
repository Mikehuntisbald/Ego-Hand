"""Verify frozen source syntax/hashes and downloaded LFS checkpoint integrity."""
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
    parser.add_argument('--code-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((REPO/'snapshot_manifest.json').read_text())
    errors = []
    for entry in manifest['sources']:
        path = REPO/entry['path']
        if not path.exists() or digest(path) != entry['sha256']:
            errors.append(f'Source missing or hash differs: {entry["path"]}')
    parsed = 0
    for path in [*sorted((REPO/'research').glob('*.py')), *sorted((REPO/'tools').glob('*.py'))]:
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path.relative_to(REPO)))
            parsed += 1
        except SyntaxError as error:
            errors.append(str(error))
    verified = 0
    if not args.code_only:
        for entry in manifest['checkpoints']:
            path = REPO/entry['path']
            if not path.exists():
                errors.append(f'Missing checkpoint: {entry["path"]}')
                continue
            if path.stat().st_size != entry['bytes']:
                errors.append(f'Size differs / LFS pointer not downloaded: {entry["path"]}')
                continue
            if digest(path) != entry['sha256']:
                errors.append(f'Checkpoint hash differs: {entry["path"]}')
                continue
            verified += 1
    print(json.dumps(dict(passed=not errors, frozen_sources=len(manifest['sources']),
                          python_files_parsed=parsed, checkpoints_verified=verified,
                          code_only=args.code_only, errors=errors), indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == '__main__':
    main()
