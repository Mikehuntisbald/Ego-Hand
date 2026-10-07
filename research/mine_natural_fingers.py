"""Natural RGB visibility scouting only. No synthetic pixel/coordinate masking."""
import json
import spatial_rgb_common as s
import numpy as np,torch,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from offline_rgb_encoder import crop_image
O=s.RUN/'natural_finger_audit';O.mkdir(exist_ok=True)
rows=json.loads((s.OLD/'rows.json').read_text());records,index=s.records_and_index();w=torch.load(s.OLD/'windows.pt',weights_only=False)
items=[]
for wi,row in enumerate(rows):
    if row['role']!='test':continue
    fid=int(index['feature_ids'][wi,8]);r=records[fid-1];xyz=np.asarray(r['gt']);ratios=[]
    for chain in [[8,9,10,1],[11,12,13,2],[14,15,16,3],[17,18,19,4]]:
        q=xyz[chain];ratios.append(np.linalg.norm(q[-1]-q[0])/max(np.linalg.norm(np.diff(q,axis=0),axis=1).sum(),1e-6))
    items.append(dict(window_index=wi,sequence=row['sequence'],clip=row['clip'],frame=row['frame'],image=r['image'],
        visible_fraction=float(r['visibility_label']),curl_ratio=float(np.mean(ratios)),projection_invalid=int((~w['valid'][wi]).sum())))
selected=[];used=set()
definitions=[('low_whole_hand_visibility',lambda q:q['visible_fraction']<.5,lambda q:q['visible_fraction']),
    ('middle_whole_hand_visibility',lambda q:.5<=q['visible_fraction']<.8,lambda q:q['visible_fraction']),
    ('curled_fingers',lambda q:q['visible_fraction']>=.5,lambda q:q['curl_ratio']),
    ('projection_or_frame_edge',lambda q:q['projection_invalid']>0,lambda q:-q['projection_invalid'])]
for group,cond,score in definitions:
    for seq in sorted({r['sequence'] for r in items}):
        options=sorted([r for r in items if r['sequence']==seq and cond(r)],key=score);clips=set();n=0
        for q in options:
            key=(q['sequence'],q['clip'])
            if key in used or key in clips:continue
            item=dict(q,scouting_group=group,id=len(selected));selected.append(item);used.add(key);clips.add(key);n+=1
            if n==2:break
for page,start in enumerate(range(0,len(selected),12)):
    part=selected[start:start+12];fig,axes=plt.subplots(3,4,figsize=(12,10),layout='constrained')
    for ax in axes.flat:ax.axis('off')
    for ax,item in zip(axes.flat,part):
        wi=item['window_index'];fid=int(index['feature_ids'][wi,8]);roi=index['roi'][fid].numpy()*1408
        im=cv2.imread(item['image']);ax.imshow(np.rot90(crop_image(im,roi)[:,:,::-1]));ax.set_title(f"ID {item['id']} | {item['sequence']}\nclip {item['clip']} f{item['frame']} | hand vis {item['visible_fraction']:.2f}",fontsize=8)
    fig.suptitle('Original RGB only | Candidate ranking is NOT per-finger visibility truth',fontsize=11)
    fig.savefig(O/f'scout_{page}.png',dpi=130);plt.close(fig)
s.save(O/'candidates.json',selected)
s.save(O/'scope.json',dict(source_windows=len(items),candidates=len(selected),natural_rgb_only=True,synthetic_pixels=False,synthetic_coordinate_gaps=False,
    per_finger_visibility_ground_truth=False,ranking='Whole-hand visibility proxy, 3D finger curl, projection invalidity; labels require visual review',
    limitations='Existing matched hand tracks from 8 previously evaluated sequences; not an exhaustive missing-detector search; projection invalidity is not occlusion'))
print(json.dumps(dict(candidates=len(selected),pages=(len(selected)+11)//12)))
