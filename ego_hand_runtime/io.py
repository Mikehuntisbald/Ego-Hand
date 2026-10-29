"""Strict JSON, atomic receipts and independently verifiable inference seals."""
import hashlib
import json
import math
import os
from pathlib import Path
from .errors import RuntimeFailure


def sha256(path):
    with Path(path).open('rb') as stream:
        if hasattr(hashlib, 'file_digest'):
            return hashlib.file_digest(stream, 'sha256').hexdigest()
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


def jsonable(value):
    if hasattr(value, 'tolist'):
        return jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical(value):
    return json.dumps(jsonable(value), sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    def bad(value):
        raise RuntimeFailure('invalid_json', 'Non-finite JSON number', value=value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=bad)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.partial')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(jsonable(value), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def seal(directory, files, *, binding, backend):
    root = Path(directory).resolve()
    names = []
    for name in files:
        target = (root / name).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            raise RuntimeFailure('seal_path', 'Seal contains a missing or escaping file')
        names.append(target.relative_to(root).as_posix())
    receipt = dict(schema=1, binding=binding, backend=backend, GT_used=False,
                   files={name: sha256(root / name) for name in sorted(names)})
    atomic_json(root / 'inference_seal.json', receipt)
    return receipt


def verify_seal(directory, expected_binding=None):
    root = Path(directory).resolve()
    receipt = read_json(root / 'inference_seal.json')
    if receipt.get('schema')!=1 or not receipt.get('files') or not receipt.get('binding'):
        raise RuntimeFailure('unsealed','Incomplete inference seal')
    if receipt.get('GT_used') is not False:
        raise RuntimeFailure('unsealed', 'Inference GT isolation is not declared')
    if expected_binding is not None and receipt.get('binding') != expected_binding:
        raise RuntimeFailure('resume_mismatch', 'Input/config/source binding differs')
    for name, digest in receipt['files'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256(path) != digest:
            raise RuntimeFailure('seal_mismatch', 'Inference output hash differs', file=name)
    return receipt
