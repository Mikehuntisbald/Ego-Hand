"""Warmup uses training GT crops; fair tune/test/fine tuning use detector crops."""
import argparse,json,os,random,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import cv2,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from coarse_pose3d import CoarsePose3D,coarse_loss
from metrics_3d import pose_metrics
ROOT=Path('/mnt/why/HOT3D');RUN=ROOT/'experiments/dit_lowconfidence_v1'

class PoseDataset(Dataset):
    def __init__(self,manifest,augment=False,roles=None):
        self.rows=[json.loads(l) for l in Path(manifest).read_text().splitlines()]
        if roles:self.rows=[r for r in self.rows if r['role'] in roles]
        self.augment=augment
        def load(row):
            image=cv2.imread(row['crop']);assert image is not None,row['crop']
            return cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        with ThreadPoolExecutor(max_workers=16) as pool:self.images=list(pool.map(load,self.rows))
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];image=self.images[i]
        if self.augment:
            gain=random.uniform(.8,1.2);offset=random.uniform(-12,12)
            image=np.clip(image.astype(np.float32)*gain+offset,0,255).astype(np.uint8)
        image=torch.from_numpy(image.transpose(2,0,1).copy()).float()/255
        geometry=torch.tensor(r['geometry'],dtype=torch.float32);gt=torch.tensor(r['xyz_camera_m'],dtype=torch.float32)
        roi=np.asarray(r['roi']);uv=(np.asarray(r['uv_pixels'])-roi[:2]+.5)/(roi[2:]-roi[:2])
        valid=np.asarray(r['projection_valid'])&(uv>=0).all(-1)&(uv<=1).all(-1)
        return dict(image=image,geometry=geometry,gt=gt,uv=torch.tensor(uv,dtype=torch.float32),valid=torch.tensor(valid,dtype=torch.float32),index=i)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['warmup','fine'],default='warmup');ap.add_argument('--epochs',type=int,default=20)
    ap.add_argument('--device',default='cuda:1');ap.add_argument('--batch',type=int,default=128);ap.add_argument('--resume');a=ap.parse_args()
    cv2.setNumThreads(0);torch.set_num_threads(4);torch.manual_seed(20261003);np.random.seed(20261003);random.seed(20261003)
    device=torch.device(a.device);folder=RUN/f'coarse_{a.phase}';folder.mkdir(exist_ok=True)
    manifest=RUN/('warmup_manifest.jsonl' if a.phase=='warmup' else 'predicted_manifest.jsonl')
    ds=PoseDataset(manifest,augment=True,roles=['coarse']);assert len(ds)>1000
    loader=DataLoader(ds,batch_size=a.batch,shuffle=True,num_workers=8,pin_memory=True,persistent_workers=True)
    tune=None
    if a.phase=='fine':
        tune=PoseDataset(manifest,roles=['tune']);assert all(r['subject']=='P0003' and r['box_source']=='predicted' for r in tune.rows)
        tune_loader=DataLoader(tune,batch_size=a.batch,num_workers=4,pin_memory=True,persistent_workers=True)
    model=CoarsePose3D().to(device)
    if a.resume:model.load_state_dict(torch.load(a.resume,map_location='cpu',weights_only=False)['model'])
    optimizer=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':.00005},{'params':[p for n,p in model.named_parameters() if not n.startswith('backbone.')],'lr':.0003}],weight_decay=.0001)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.epochs,eta_min=.000005)
    best=float('inf');started=time.time();history=[]
    for epoch in range(a.epochs):
        model.train();total=0.;n=0
        for batch in loader:
            batch={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):pred=model(batch['image'],batch['geometry'])
            pred={k:v.float() for k,v in pred.items()};loss,parts=coarse_loss(pred,batch['gt'],batch['uv'],batch['valid'])
            assert torch.isfinite(loss),parts;loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();total+=float(loss.detach())*len(batch['gt']);n+=len(batch['gt'])
        scheduler.step();row=dict(epoch=epoch+1,train_loss=total/n,seconds=time.time()-started,samples=n)
        if tune:
            model.eval();pp=[];gg=[]
            with torch.inference_mode():
                for batch in tune_loader:
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):p=model(batch['image'].to(device),batch['geometry'].to(device))
                    pp.append(p['xyz'].float().cpu().numpy());gg.append(batch['gt'].numpy())
            metric=pose_metrics(np.concatenate(pp),np.concatenate(gg));row.update(tune={k:v for k,v in metric.items() if not k.startswith('sample')})
            score=metric['mpjpe19_mm']+metric['wrist_relative_mpjpe19_mm']
        else:score=total/n
        checkpoint=dict(model=model.state_dict(),epoch=epoch+1,phase=a.phase,history=history+[row],model_definition='YOLO26s backbone + calibrated canonical20 pose')
        torch.save(checkpoint,folder/'last.pt')
        if score<best:best=score;torch.save(checkpoint,folder/'best.pt')
        history.append(row);(folder/'history.json').write_text(json.dumps(history,indent=2));print(json.dumps(row),flush=True)
    (folder/'done.json').write_text(json.dumps(dict(completed=True,checkpoint=str(folder/'best.pt'),phase=a.phase)))

if __name__=='__main__':main()
