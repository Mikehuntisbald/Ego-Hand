"""Predicted masks are inputs; original human polygons are loss-only targets."""
import argparse,collections,json,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from instance_masks_v51 import HandInstanceSegmenter
from cache_instance_conditions_v51 import RUN

def crop_array(image,box):
    box=np.asarray(box,float);cx,cy=(box[:2]+box[2:])/2
    edge=max((box[2]-box[0])*256/192,box[3]-box[1])*1.3
    yy,xx=np.mgrid[:256,:256];mx=(cx+(xx-127.5)*edge/256).astype(np.float32);my=(cy+(yy-127.5)*edge/256).astype(np.float32)
    return cv2.remap(image,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)

def cells(mask):
    # Exactly the kernel16/stride16/padding2 native patch embedding support.
    x=torch.from_numpy(np.rot90(mask,1).copy()).float()[None,None,: ,32:-32]
    return F.avg_pool2d(x,16,16,padding=2,count_include_pad=False).flatten().numpy()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');ap.add_argument('--per-group',type=int,default=4);a=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(0)
    if (RUN/'domain_masks_done.json').exists():return
    rows=[json.loads(x) for x in (RUN/'domain_records.jsonl').read_text().splitlines()]
    groups=collections.defaultdict(list)
    for row in rows:
        if row['dataset']=='egohands' and row['split']!='test':groups[row['group']].append(row)
    chosen=[]
    for group,records in sorted(groups.items()):
        records.sort(key=lambda r:r['frame_index']);ids=np.linspace(0,len(records)-1,min(a.per_group,len(records))).round().astype(int)
        chosen.extend(records[i] for i in ids)
    segmenter=HandInstanceSegmenter(a.device);inputs=[];targets=[];metadata=[];qualities=[];ious=[];pixels=[];start=time.time()
    for n,row in enumerate(chosen):
        image=cv2.imread(row['image']);boxes=[h['box_xyxy'] for h in row['hands']]
        predicted=segmenter(image,boxes)
        # Predictions above are complete before the separate human label read.
        with np.load(row['mask_gt']) as f:gt=f['masks']
        for hand,pr in zip(row['hands'],predicted):
            box=hand['box_xyxy'];mask=gt[hand['mask_index']]
            pixels.append(crop_array(image,box));inputs.append(np.stack([cells(crop_array(pr['mask'].astype(np.float32),box)),cells(crop_array(pr['other'].astype(np.float32),box))]))
            targets.append(cells(crop_array(mask.astype(np.float32),box)));qualities.append(pr['quality'])
            iou=float((pr['mask']&mask.astype(bool)).sum()/max((pr['mask']|mask.astype(bool)).sum(),1));ious.append(iou)
            metadata.append(dict(id=row['id'],split=row['split'],group=row['group'],instance=hand['instance_id'],sam_iou=iou,box_source='human_train_or_dev_ROI; conditional segmentation diagnostic only'))
        if n%10==0:print(json.dumps(dict(stage='domain_masks',done=n+1,total=len(chosen),seconds=time.time()-start)),flush=True)
    np.save(RUN/'domain_pixels.npy',np.stack(pixels))
    torch.save(dict(conditions=torch.from_numpy(np.stack(inputs)),target=torch.from_numpy(np.stack(targets)),quality=torch.tensor(qualities),metadata=metadata),RUN/'domain_masks.pt')
    (RUN/'domain_masks_done.json').write_text(json.dumps(dict(complete=True,images=len(chosen),instances=len(inputs),mean_sam_IoU=float(np.mean(ious)),GT_only_in_loss_targets=True,GT_3D=False,conditional_on_annotated_ROIs=True,no_test_images_read=True,seconds=time.time()-start),indent=2))

if __name__=='__main__':main()
