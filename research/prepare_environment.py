"""Install small missing dependencies only in HOT3D's isolated environment."""
import hashlib, importlib.util, json, subprocess, sys
from pathlib import Path
import requests

ROOT = Path('/mnt/why/HOT3D')
packages = {'webdataset':'webdataset', 'braceexpand':'braceexpand', 'polars':'polars', 'polars-runtime-32':'_polars_runtime_32', 'ultralytics_thop':'thop'}
missing = [package for package, module in packages.items() if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '-i', 'https://pypi.tuna.tsinghua.edu.cn/simple', *missing], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-build-isolation', '-e', '/mnt/why/HOT3D-hand-tracking-toolkit'], check=True)
sys.path.insert(0, '/mnt/why/HOT3D-hand-tracking-toolkit')
import torch, cv2, ultralytics
from hand_tracking_toolkit import dataset
meta = requests.get('https://hf-mirror.com/api/models/Ultralytics/YOLO26', timeout=30).json()
revision = meta['sha']
tree = requests.get(f'https://hf-mirror.com/api/models/Ultralytics/YOLO26/tree/{revision}', timeout=30).json()
tree = {item['path']:item for item in tree}
(ROOT / 'weights').mkdir(exist_ok=True)
receipts = []
for name in ['yolo26s.pt', 'yolo26s-pose.pt']:
    entry = tree[name]
    dest = ROOT / 'weights' / name
    expected = entry['lfs']['oid']
    if not dest.exists() or hashlib.sha256(dest.read_bytes()).hexdigest() != expected:
        r = requests.get(f'https://hf-mirror.com/Ultralytics/YOLO26/resolve/{revision}/{name}', timeout=(15,90), stream=True)
        r.raise_for_status()
        part = dest.with_suffix('.pt.partial')
        with part.open('wb') as output:
            for chunk in r.iter_content(1 << 20): output.write(chunk)
        assert hashlib.sha256(part.read_bytes()).hexdigest() == expected, name
        part.replace(dest)
    receipts.append(dict(name=name, revision=revision, sha256=expected, bytes=dest.stat().st_size))
result = dict(python=sys.version, torch=torch.__version__, opencv=cv2.__version__, ultralytics=ultralytics.__version__, weights=receipts)
(ROOT/'provenance/environment.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2),flush=True)
