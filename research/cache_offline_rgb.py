import argparse,json,hashlib,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import wilor_eval_common as common
import cv2,numpy as np,torch
from offline_rgb_data import RUN,save
from offline_rgb_encoder import RGBEncoder,WEIGHT,crop_roi,crop_image,rectangles,cover,input_tensor

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--device',default='cuda:0');a=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(0)
    if (RUN/'rgb_done.json').exists():return
    rows=json.loads((RUN/'rows.json').read_text());data=torch.load(RUN/'windows.pt',weights_only=False)
    source_rows={s:json.loads((common.ROOT/'experiments'/s/'locked_rows.json').read_text()) for s in sorted({r['source'] for r in rows})}
    keys=sorted({(r['source'],int(j)) for i,r in enumerate(rows) for j in data['obs_ids'][i].tolist() if j>=0})
    lookup={k:i+1 for i,k in enumerate(keys)};records=[dict(source_rows[s][j],source=s,index=j) for s,j in keys]
    feature_ids=torch.zeros(len(rows),17,dtype=torch.long)
    for i,r in enumerate(rows):
        for k,j in enumerate(data['obs_ids'][i].tolist()):
            if j>=0:feature_ids[i,k]=lookup[(r['source'],j)]
    rois=np.zeros((len(keys)+1,4),np.float32);rects=np.zeros((len(keys)+1,6,4),np.int64)
    for i,r in enumerate(records,1):rois[i]=crop_roi(r['box']);rects[i]=rectangles(r['sequence'],r['clip'])
    save(RUN/'rgb_records.json',records)
    torch.save(dict(feature_ids=feature_ids,roi=torch.from_numpy(rois/1408),rectangles=torch.from_numpy(rects)),RUN/'rgb_index.pt')
    policy=dict(version='rgb_v2',modalities='RGB+tracks vs tracks-only, same geometry and temporal masks',
        pixels='Native RGB cropped with supplied YOLO box, opaque rectangles applied BEFORE frozen encoder; no unmasked feature mixing within affected frames',
        keypoints='ALL keypoint observations removed from every affected frame, eliminating old unmasked predictions as an indirect input',
        crop='1.4x square YOLO hand box; current hand tracking/detection assumed available; no claim of full hand-box recovery',
        mask_geometry='Clip-metadata-seeded four crop-edge rectangles, plus full crop; independent of GT and predicted finger coordinates',
        encoder='Frozen HOT3D YOLO26 backbone layers 0..6; concatenated layer4/layer6 4x4 maps, 16 tokens x 512 channels',
        encoder_sha256=hashlib.sha256(WEIGHT.read_bytes()).hexdigest(),
        test='P0010/P0015 sequences from v3, not v1 keypoint test v4; previously opened in older 3D studies, not untouched research data',
        natural='Unmodified naturally occluded RGB with independent coordinate gaps; whole-hand visibility grouping is diagnostic, NOT per-finger occlusion truth')
    save(RUN/'rgb_protocol.json',policy)
    model=RGBEncoder().to(a.device).eval();folder=RUN/'rgb_chunks';folder.mkdir(exist_ok=True)
    def prepare(i):
        r=records[i];image=cv2.imread(r['image']);assert image is not None
        crop=crop_image(image,rois[i+1]);color=32+64*((r['clip']+1)%4)
        return [cover(crop,rect,color) for rect in rects[i+1]]
    started=time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0,len(records),256):
            end=min(start+256,len(records));dest=folder/f'{start:06d}.pt'
            if dest.exists():continue
            prepared=list(pool.map(prepare,range(start,end)));flat=[im for variant in prepared for im in variant];out=[]
            for b in range(0,len(flat),96):
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):features=model(input_tensor(flat[b:b+96],a.device))
                out.append(features.float().half().cpu())
            values=torch.cat(out).reshape(end-start,6,16,512);assert torch.isfinite(values).all()
            tmp=dest.with_suffix('.partial');torch.save(dict(start=start,end=end,features=values),tmp);tmp.replace(dest)
            print(json.dumps(dict(stage='masked_rgb_cache',done=end,total=len(records),seconds=time.time()-started)),flush=True)
    save(RUN/'rgb_done.json',dict(complete=True,observations=len(records),variants=6,gt_used_for_crops_or_masks=False))

if __name__=='__main__':main()
