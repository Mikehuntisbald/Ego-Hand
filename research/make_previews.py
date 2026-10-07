"""Select subject-disjoint validation GT overlays, not model predictions."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from export_hand_labels import EDGES
root=Path('/mnt/why/HOT3D');out=root/'previews';out.mkdir(exist_ok=True)
manifest=json.loads((root/'subset_manifest.json').read_text());selected=[]
for subject in manifest['validation_subjects']:
    candidates=[]
    files=sorted((root/'export/annotations/val').glob(f'{subject}_*/clip-*.jsonl'))[:5]
    for path in files:
        for line in path.read_text().splitlines()[::5]:
            row=json.loads(line)
            for hand in row['hands']:
                v=hand['modeled_hand_visible_fraction'];box=hand['box_amodal_xyxy']
                if v is None or box is None or sum(hand['keypoint_projection_valid'])<17:continue
                if box[2]-box[0]<80 or box[3]-box[1]<80:continue
                candidates.append((abs(v-.4),row,hand['side']))
    if not candidates:continue
    _,row,chosen=min(candidates,key=lambda c:c[0])
    img=Image.open(root/'export'/row['image']).convert('RGB');draw=ImageDraw.Draw(img)
    for hand in row['hands']:
        color='#22dd88' if hand['side']=='left' else '#ff9955'
        if hand['box_amodal_xyxy']:draw.rectangle(hand['box_amodal_xyxy'],outline=color,width=4)
        uv=np.asarray(hand['uv_pixels']);valid=hand['keypoint_projection_valid']
        for a,b in EDGES:
            if valid[a] and valid[b]:draw.line([tuple(uv[a]),tuple(uv[b])],fill=color,width=4)
        for (x,y),v in zip(uv,valid):
            if v:draw.ellipse((x-5,y-5,x+5,y+5),fill=color)
    name=f"{subject}-{row['clip']:06d}-{row['frame']:06d}-gt.jpg"
    img.resize((704,704)).save(out/name,quality=90)
    chosen_hand=next(h for h in row['hands'] if h['side']==chosen)
    selected.append(dict(subject=subject,split='val',sequence=row['sequence'],clip=row['clip'],frame=row['frame'],
        modeled_hand_visible_fraction=chosen_hand['modeled_hand_visible_fraction'],chosen_hand=chosen,
        file=name,caption='GT overlay; modeled whole-hand visibility, not observed per-keypoint visibility'))
(out/'index.json').write_text(json.dumps(selected,indent=2));print(json.dumps(selected,indent=2))
