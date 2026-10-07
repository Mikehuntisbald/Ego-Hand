import json
import spatial_rgb_common as s
import numpy as np,cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from offline_rgb_encoder import crop_image
O=s.RUN/'natural_finger_audit';items=json.loads((O/'candidates.json').read_text());records,index=s.records_and_index()
notes={12:('object_occlusion','Toy house hides substantial finger segments; only part of the grasp contour remains.'),
6:('object_occlusion','Mug hides finger segments; thumb and wrist give only partial constraints.'),
25:('inter_hand_and_object','Two hands and a can overlap; finger identity and boundaries need temporal review.'),
33:('self_occlusion','Flexed fingers: back of hand is visible but distal segments/tips overlap or turn away. Whole-hand visibility=1 is insufficient.'),
37:('motion_blur','Motion blur obscures joint boundaries despite high whole-hand visibility; not equivalent to opaque occlusion.'),
45:('frame_edge_and_dark','Hand approaches image boundary and is dark; confirm out-of-view separately from within-image occlusion.'),
28:('object_occlusion','Juice carton hides grasping finger segments while thumb/palm remain partly visible.'),
36:('grasp_self_and_object','Grasp around banana-shaped object hides distal finger surfaces despite high whole-hand visibility.')}
selected=[]
for page,ids in enumerate([[12,6,25,33],[37,45,28,36]]):
    fig,axes=plt.subplots(4,4,figsize=(13,12),layout='constrained')
    for rr,id in enumerate(ids):
        item=items[id];wi=item['window_index'];tag,note=notes[id];available=[]
        for cc,slot in enumerate([5,8,11]):
            ax=axes[rr,cc];ax.axis('off');fid=int(index['feature_ids'][wi,slot])
            if not fid:ax.text(.1,.5,'No associated frame');continue
            r=records[fid-1];roi=index['roi'][fid].numpy()*1408;im=cv2.imread(r['image'])
            ax.imshow(np.rot90(crop_image(im,roi)[:,:,::-1]));ax.set_title(['Past original RGB','Current original RGB','Future original RGB'][cc],fontsize=9)
            available.append(dict(slot=slot,image=r['image'],box=r['box']))
        fid=int(index['feature_ids'][wi,8]);r=records[fid-1];im=cv2.imread(r['image']);roi=index['roi'][fid].numpy()*1408
        xy=np.round(roi).astype(int);cv2.rectangle(im,(xy[0],xy[1]),(xy[2],xy[3]),(0,230,230),5)
        axes[rr,3].imshow(np.rot90(im[:,:,::-1]));axes[rr,3].axis('off');axes[rr,3].set_title('Full original frame + crop outline',fontsize=9)
        axes[rr,1].set_axis_on();axes[rr,1].set_xticks([]);axes[rr,1].set_yticks([])
        axes[rr,1].set_xlabel(f"ID {id}: {tag}\n{item['sequence']} clip {item['clip']} frame {item['frame']}",fontsize=8)
        selected.append(dict(**item,visual_review_category=tag,visual_review_note=note,label_status='assistant visual review, not ground truth or per-joint visibility annotation',context=available))
    fig.suptitle('Natural finger visibility review | NO synthetic occlusion or coordinate removal',fontsize=12)
    fig.savefig(O/f'review_{page}.png',dpi=140);plt.close(fig)
s.save(O/'reviewed_examples.json',selected)
print(json.dumps(dict(reviewed_candidates=len(items),illustrated_examples=len(selected))))
