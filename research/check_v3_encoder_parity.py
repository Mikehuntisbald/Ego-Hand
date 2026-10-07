"""Check exported observation encoder against the actual training cache."""
import json
import sys
from pathlib import Path
RUN=Path('/mnt/why/HOT3D/experiments/dit_wilor_v3')
sys.path.insert(0,str(RUN/'sealed'))
import cv2
import numpy as np
import torch
from dit_v3_inference import ObservationEncoder
from cache_dit_v3 import observed_crop
from compare_detectors import iou

torch.set_num_threads(4);cv2.setNumThreads(0)
rows=json.loads((RUN/'rows.json').read_text());sides=json.loads((RUN/'side_predictions.json').read_text())
reference=torch.load(RUN/'feature_chunks/000000.pt',map_location='cpu',weights_only=False)
prepared=[]
for r in rows[:16]:
    path=Path(r['image']);split=path.parent.parent.parent.name
    ann=Path('/mnt/why/HOT3D/export/annotations')/split/r['sequence']/f'clip-{r["clip"]:06d}.jsonl'
    camera=json.loads(ann.read_text().splitlines()[r['frame']])['camera'];side=sides[r['image']]
    overlap=iou([r['roi']],side['boxes'])[0]
    if len(overlap) and overlap.max()>=.05:
        j=int(overlap.argmax());right=int(side['classes'][j]);conf=float(side['scores'][j]);ov=float(overlap[j])
    else:right=1;conf=0.;ov=0.
    crop,R,f,e,fb=observed_crop(cv2.imread(r['image']),r['roi'],camera)
    prepared.append((crop,R,f,right,conf,ov,e,fb))
encoder=ObservationEncoder('cuda:0')
out=encoder.features(*map(list,zip(*prepared)))
checks={}
for key in ['global','local','wilor','transform','geometry','wilor_2d','shape']:
    x=out[key].float().cpu();y=reference[key][:16].float()
    error=float((x-y).abs().max());checks[key]=error
    assert torch.allclose(x,y,atol=.002 if key in ['global','local'] else 1e-5,rtol=.001),key
(RUN/'encoder_parity.json').write_text(json.dumps(dict(passed=True,max_absolute_errors=checks),indent=2))
print(json.dumps(checks,indent=2))
