"""A fold is initialized from COCO only and never trains on its held subject."""
import argparse,json,random,time
import cv2,numpy as np,torch
from torch.utils.data import DataLoader
from train_coarse_pose import PoseDataset
from coarse_pose3d import CoarsePose3D,coarse_loss
from metrics_3d import pose_metrics
from oof_common import OLD,RUN,SUBJECTS,save,sha

def train_phase(subject,phase,device):
    folder=RUN/'folds'/subject/phase
    if (folder/'done.json').exists():return
    folder.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(20261003);np.random.seed(20261003);random.seed(20261003)
    manifest=OLD/('warmup_manifest.jsonl' if phase=='warmup' else 'predicted_manifest.jsonl')
    ds=PoseDataset(manifest,augment=True,roles=['coarse'])
    keep=[i for i,r in enumerate(ds.rows) if r['subject']!=subject]
    ds.rows=[ds.rows[i] for i in keep];ds.images=[ds.images[i] for i in keep]
    assert {r['subject'] for r in ds.rows}==set(SUBJECTS)-{subject}
    loader=DataLoader(ds,batch_size=128,shuffle=True,num_workers=4,pin_memory=True,persistent_workers=True)
    tune=PoseDataset(OLD/'predicted_manifest.jsonl',roles=['tune']) if phase=='fine' else None
    if tune:
        assert {r['subject'] for r in tune.rows}=={'P0003'}
        val=DataLoader(tune,batch_size=128,num_workers=2,pin_memory=True,persistent_workers=True)
    model=CoarsePose3D().to(device)
    if phase=='fine':model.load_state_dict(torch.load(folder.parent/'warmup/best.pt',map_location='cpu',weights_only=False)['model'])
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':.00005},
        {'params':[p for n,p in model.named_parameters() if not n.startswith('backbone.')],'lr':.0003}],weight_decay=.0001)
    epochs=20 if phase=='warmup' else 40
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,epochs,eta_min=.000005)
    best=float('inf');history=[];started=time.time()
    provenance=dict(held_subject=subject,trained_subjects=sorted(set(SUBJECTS)-{subject}),
        initialization='COCO YOLO26s only; no v1 3D checkpoint',selection_subject='P0003' if tune else None,
        train_manifest_sha256=sha(manifest),phase=phase,samples=len(ds))
    for epoch in range(1,epochs+1):
        model.train();total=0.;n=0
        for b in loader:
            b={k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in b.items()}
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model(b['image'],b['geometry'])
            loss,parts=coarse_loss({k:v.float() for k,v in p.items()},b['gt'],b['uv'],b['valid'])
            assert torch.isfinite(loss),parts
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total+=float(loss.detach())*len(b['gt']);n+=len(b['gt'])
        sched.step();row=dict(epoch=epoch,train_loss=total/n,seconds=time.time()-started)
        score=total/n
        if tune:
            model.eval();pp=[];gg=[]
            with torch.inference_mode():
                for b in val:
                    with torch.autocast('cuda',dtype=torch.bfloat16):p=model(b['image'].to(device),b['geometry'].to(device))
                    pp.append(p['xyz'].float().cpu().numpy());gg.append(b['gt'].numpy())
            metrics=pose_metrics(np.concatenate(pp),np.concatenate(gg));metrics.pop('sample_mpjpe19_mm')
            row['tune']=metrics;score=metrics['mpjpe19_mm']+metrics['wrist_relative_mpjpe19_mm']
        history.append(row)
        if score<best:
            best=score;torch.save(dict(model=model.state_dict(),epoch=epoch,provenance=provenance),folder/'best.pt')
        save(folder/'history.json',history);print(json.dumps(dict(subject=subject,phase=phase,**row)),flush=True)
    save(folder/'done.json',dict(completed=True,checkpoint=str(folder/'best.pt'),sha256=sha(folder/'best.pt'),**provenance))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--subjects',nargs='+',required=True);ap.add_argument('--device',required=True);a=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(0)
    for subject in a.subjects:
        assert subject in SUBJECTS
        train_phase(subject,'warmup',a.device);train_phase(subject,'fine',a.device)
if __name__=='__main__':main()

