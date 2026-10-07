from pathlib import Path
import json, tarfile
root=Path('/mnt/why/HOT3D-hand-tracking-toolkit/hand_tracking_toolkit')
for name in ['visualization.py','camera.py']:
    text=(root/name).read_text().splitlines()
    print(name)
    if name=='visualization.py':print('\n'.join(text[:160]))
    else:
        for i,l in enumerate(text):
            if 'def from_json' in l or 'def eye_to_window' in l or 'def world_to_eye' in l: print('\n'.join(text[i:i+45]))
with tarfile.open('/mnt/why/HOT3D/rgb_clips/train/P0001_9b6feab7/clip-001892.tar') as t:
    for suffix in ['.hands.json','.cameras.json']:
        name=next(m.name for m in t if m.name.endswith(suffix))
        print(name,json.loads(t.extractfile(name).read()))
