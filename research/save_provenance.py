import hashlib, importlib.metadata, json, subprocess
from pathlib import Path
import requests
root=Path('/mnt/why/HOT3D');p=root/'provenance';m=json.loads((root/'subset_manifest.json').read_text())
url=f"https://hf-mirror.com/datasets/{m['official_repo']}/resolve/{m['revision']}/README.md"
r=requests.get(url,timeout=30);r.raise_for_status();(p/'official_dataset_README.md').write_bytes(r.content)
versions={}
for name in ['torch','numpy','opencv-python','opencv-python-headless','Pillow','requests','ultralytics','webdataset','braceexpand','polars','polars-runtime-32','ultralytics-thop','hand_tracking_toolkit']:
    try:versions[name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:pass
import cv2
versions['cv2_module']=cv2.__version__
repos={name:subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True).strip() for name,folder in [('hot3d',Path('/mnt/why/HOT3D-toolkit')),('hand_tracking_toolkit',Path('/mnt/why/HOT3D-hand-tracking-toolkit'))]}
sources=dict(official_clips_repo='https://huggingface.co/datasets/bop-benchmark/hot3d',official_toolkit='https://github.com/facebookresearch/hot3d',
    hand_toolkit='https://github.com/facebookresearch/hand_tracking_toolkit',yolo26='https://docs.ultralytics.com/models/yolo26/',
    dit='https://github.com/facebookresearch/DiT',modelscope_repo='https://modelscope.cn/datasets/projectaria/hot3d',
    modelscope_probe='Repository contains description and example assets, not sequence payloads',
    mirror_entry='https://hf-mirror.com',binary_redirect_host='cas-bridge.xethub.hf.co',
    mirror_note='Chinese mirror entry; payload redirect is not a guarantee of a China-hosted CDN',
    revisions=repos,clip_revision=m['revision'],versions=versions,
    manifest_sha256=hashlib.sha256((root/'subset_manifest.json').read_bytes()).hexdigest())
(p/'sources_and_versions.json').write_text(json.dumps(sources,indent=2));print(json.dumps(sources,indent=2))
