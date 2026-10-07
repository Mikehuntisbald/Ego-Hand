"""Cache actual frozen 3D predictions and image features; GT remains label tensors."""
import json,torch
from torch.utils.data import DataLoader
from train_coarse_pose import PoseDataset,RUN
from coarse_pose3d import CoarsePose3D

def main():
    torch.set_num_threads(4);device='cuda:1';ds=PoseDataset(RUN/'predicted_manifest.jsonl')
    model=CoarsePose3D().to(device);ckpt=torch.load(RUN/'coarse_fine/best.pt',map_location='cpu',weights_only=False);model.load_state_dict(ckpt['model']);model.eval()
    loader=DataLoader(ds,batch_size=256,num_workers=8,pin_memory=True,persistent_workers=True)
    output={k:[] for k in ['coarse','confidence','rgb','gt']}
    with torch.inference_mode():
        for b in loader:
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(b['image'].to(device),b['geometry'].to(device))
            output['coarse'].append(p['xyz'].float().cpu());output['rgb'].append(p['rgb_tokens'].half().cpu());output['gt'].append(b['gt'])
            output['confidence'].append(torch.cat([p['root_confidence'][:,None],p['confidence']],1).float().cpu())
    data={k:torch.cat(v) for k,v in output.items()};data['rows']=ds.rows
    torch.save(data,RUN/'coarse_cache.pt')
    (RUN/'cache_done.json').write_text(json.dumps(dict(completed=True,samples=len(ds),feature_shape=list(data['rgb'].shape),coarse_epoch=ckpt['epoch'])))
    print((RUN/'cache_done.json').read_text(),flush=True)

if __name__=='__main__':main()
