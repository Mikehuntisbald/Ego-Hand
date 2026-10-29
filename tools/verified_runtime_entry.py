"""Image entry: verify baked source bytes before importing any inference backend."""
from pathlib import Path
import os
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / 'runtime_source_sha256.json'
    check = subprocess.run([sys.executable, str(root/'tools/verify_runtime_release.py'),
                            '--manifest', str(manifest), '--root', str(root)],
                           capture_output=True, text=True)
    if check.returncode:
        sys.stderr.write('Runtime source verification failed before model import.\n')
        sys.stderr.write(check.stdout + check.stderr)
        return check.returncode
    sys.stderr.write('Runtime baked source manifest verified.\n')
    os.chdir(root)
    os.execv(sys.executable, [sys.executable, '-m', 'ego_hand_runtime', *sys.argv[1:]])


if __name__ == '__main__':
    raise SystemExit(main())
