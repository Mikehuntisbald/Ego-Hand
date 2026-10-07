import json
import wilor_eval_common
import numpy as np
import torch
from pathlib import Path
from offline_kp_data import RUN,save
from offline_kp_model import artificial_gap

rows=json.loads((RUN/'rows.json').read_text());data=torch.load(RUN/'windows.pt',weights_only=False)
# Metadata-only selection, before inspecting completion error.
idx=next(i for i,r in enumerate(rows) if r['role']=='test' and 40<=r['frame']<=105)
fingers=torch.zeros(1,5,dtype=torch.bool);fingers[:,1]=True
b,_=artificial_gap(data['xy'][idx:idx+1],data['observed'][idx:idx+1],data['dt'][idx:idx+1],torch.tensor([6]),fingers)
row=rows[idx];frames=[];folder=RUN/'delivery';folder.mkdir(exist_ok=True)
for k in range(17):
    x=b['xy'][0,k].numpy()*1408;seen=b['observed'][0,k].numpy()
    image=Path(row['image']).parent/f"{row['frame']+(k-8)*5:06d}.jpg"
    frames.append(dict(timestamp_s=float(b['dt'][0,k])+10,xy_px=[p.tolist() if s else None for p,s in zip(x,seen)],observed=seen.tolist(),image=str(image)))
save(folder/'example_input.json',dict(image_size=[1408,1408],tracks=[dict(id='example',frames=frames)]))
save(folder/'example_provenance.json',dict(row_index=idx,selection='First test row with a full 17-frame context by metadata, not selected by error',
    masked_finger='index',gap_frames=6,input_gt_used=False,images_for_review_only=True))
print(str(folder/'example_input.json'))
