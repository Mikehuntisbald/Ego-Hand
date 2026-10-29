"""Verify a NEW runtime manifest without editing the frozen research manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--root');args=p.parse_args()
    manifest=Path(args.manifest).resolve();root=Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    data=json.loads(manifest.read_text(encoding='utf-8-sig'));bad=[]
    for name,value in data['files'].items():
        target=(root/name).resolve()
        if not target.is_relative_to(root) or not target.is_file() or digest(target)!=value:bad.append(name)
    if not data['files'] or bad:
        print(json.dumps(dict(passed=False,failures=bad)));return 1
    print(json.dumps(dict(passed=True,files=len(data['files']),scope=data['scope'])));return 0


if __name__=='__main__':raise SystemExit(main())
