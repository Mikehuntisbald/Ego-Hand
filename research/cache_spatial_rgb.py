import argparse,json,time,hashlib
from concurrent.futures import ThreadPoolExecutor
import spatial_rgb_common as s
import wilor_eval_common as common
import cv2,numpy as np,torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:0');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=3);p.add_argument('--limit',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(0);s.RUN.mkdir(exist_ok=True)
    records,index=s.records_and_index();folder=s.RUN/'dense_chunks';folder.mkdir(exist_ok=True)
    full,_=common.load_model(a.device);model=full.backbone;del full
    assert len(model.blocks)==32 and not model.skip_blocks
    patch=model.patch_embed.proj
    assert patch.kernel_size==(16,16) and patch.stride==(16,16) and patch.padding==(2,2)
    started=time.time();finished=0
    def prep(i):
        # Explicit input whitelist excludes all stored annotation fields.
        r=records[i];observed={k:r[k] for k in ['image','camera','clip']}
        return s.prepare(observed,index['roi'][i+1].numpy()*1408,index['rectangles'][i+1].numpy())
    with ThreadPoolExecutor(max_workers=6) as pool,torch.inference_mode():
        for start in range(a.shard*64,len(records),a.shards*64):
            end=min(start+64,len(records));dest=folder/f'{start:06d}.pt'
            if dest.exists():continue
            prepared=list(pool.map(prep,range(start,end)))
            crops,positions,transform,focal,error,fallback=map(list,zip(*prepared))
            flat=[im for variants in crops for im in variants];features=[]
            for b in range(0,len(flat),16):
                images=flat[b:b+16];n=len(images);images=images+[images[-1]]*(16-n)
                x=common.input_tensor(images,[1]*16,rotation=1).to(a.device)
                with torch.autocast('cuda',dtype=torch.bfloat16):out=model(x[:,:,:,32:-32])[-1]
                assert out.shape[1:]==(1280,16,12)
                features.append(out[:n].flatten(2).transpose(1,2).half().cpu())
            features=torch.cat(features).reshape(end-start,6,192,1280)
            assert torch.isfinite(features).all()
            result=dict(start=start,end=end,features=features,positions=torch.from_numpy(np.stack(positions)),
                transform=torch.from_numpy(np.stack(transform)),focal=torch.tensor(focal),reprojection_error=error,fallback=fallback)
            tmp=dest.with_suffix('.partial');torch.save(result,tmp);tmp.replace(dest);finished+=1
            print(json.dumps(dict(shard=a.shard,start=start,end=end,seconds=time.time()-started)),flush=True)
            if a.limit and finished>=a.limit:return
    s.save(s.RUN/f'cache_shard{a.shard}.json',dict(complete=True,shards=a.shards,seconds=time.time()-started))

if __name__=='__main__':main()
