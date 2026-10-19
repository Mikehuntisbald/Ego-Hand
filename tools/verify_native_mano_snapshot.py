"""Verify the native MANO research snapshot without importing GPU libraries."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / 'evidence/native_mano_h20_20261019/manifest.json'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def verify(code_only=False):
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    errors, sources, checkpoints = [], 0, 0
    for entry in manifest['files']:
        if code_only and entry['kind'] == 'checkpoint':
            continue
        path = REPO / entry['path']
        if not path.is_file() or path.stat().st_size != entry['bytes']:
            errors.append('Missing or size differs (possibly an LFS pointer): ' + entry['path'])
            continue
        if digest(path) != entry['sha256']:
            errors.append('SHA differs: ' + entry['path'])
            continue
        if path.suffix == '.py':
            try:
                ast.parse(path.read_text(encoding='utf-8-sig'), filename=entry['path'])
                sources += 1
            except SyntaxError as error:
                errors.append(str(error))
        checkpoints += entry['kind'] == 'checkpoint'
    return dict(passed=not errors, sources_parsed=sources,
                checkpoints_verified=checkpoints, code_only=code_only,
                GPU_execution_checked=False, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-only', action='store_true')
    args = parser.parse_args()
    result = verify(args.code_only)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
