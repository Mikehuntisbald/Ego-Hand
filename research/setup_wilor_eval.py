import importlib.util
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT=Path('/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003')
with zipfile.ZipFile(ROOT/'assets/wilor_source.zip') as z:
    assert all('..' not in Path(n).parts and not n.startswith('/') for n in z.namelist())
    z.extractall(ROOT/'source')
source=next(p for p in (ROOT/'source').iterdir() if p.is_dir())
(source/'mano_data/mano').mkdir(parents=True,exist_ok=True)
shutil.copy2(ROOT/'assets/MANO_RIGHT.pkl',source/'mano_data/mano/MANO_RIGHT.pkl')
shutil.copy2(ROOT/'assets/mano_mean_params.npz',source/'mano_data/mano_mean_params.npz')
modules=['torch','torchvision','pytorch_lightning','smplx','chumpy','timm','einops','yacs','pyrender','skimage','omegaconf','hydra','pyrootutils','rich','webdataset','dill']
print(json.dumps(dict(python=sys.executable,source=str(source),modules={n:bool(importlib.util.find_spec(n)) for n in modules}),indent=2))
import hand_tracking_toolkit
print('SDK',hand_tracking_toolkit.__file__)
(ROOT/'source_path.txt').write_text(str(source))
