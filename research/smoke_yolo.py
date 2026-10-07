"""Check hand dataset decoding and both custom YOLO26 heads with real GT."""
import json, time
from pathlib import Path
import torch, yaml
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.data.dataset import YOLODataset
from ultralytics.nn.tasks import DetectionModel, PoseModel

root=Path('/mnt/why/HOT3D');results=[];torch.set_num_threads(4)
def finite(value):
    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite(v) for v in value.values())
    if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
    return True
def serial(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
    if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [serial(v) for v in value]
    return value
for task in ['detect','pose']:
    data=yaml.safe_load((root/'export'/task/'data.yaml').read_text())
    listing=root/'export'/task/'val.txt'
    if not listing.exists() or not listing.read_text().strip():listing=root/'export'/task/'train.txt'
    ds=YOLODataset(img_path=str(listing),imgsz=640,batch_size=2,augment=False,
                   hyp=get_cfg(),data=data,task=task,cache=False)
    assert ds.ni>0
    assert sum(len(l['cls']) for l in ds.labels)>0,'Labels lost while resolving shared RGB paths'
    batch=ds.collate_fn([ds[0],ds[min(1,len(ds)-1)]])
    batch={k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in batch.items()}
    batch['img']=batch['img'].float()/255
    pretrained=YOLO(str(root/'weights'/('yolo26s-pose.pt' if task=='pose' else 'yolo26s.pt')))
    if task=='pose':
        model=PoseModel(cfg='yolo26s-pose.yaml',ch=3,nc=1,data_kpt_shape=(20,3),verbose=False)
        assert tuple(model.model[-1].kpt_shape)==(20,3)
        assert batch['keypoints'].shape[-2:]==(20,3)
    else:model=DetectionModel(cfg='yolo26s.yaml',ch=3,nc=1,verbose=False)
    model.load(pretrained.model,verbose=False);model.args=get_cfg();model.to('cuda:0').train()
    started=time.time();loss,components=model(batch)
    assert finite(loss) and finite(components)
    loss.sum().backward()
    grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    results.append(dict(task=task,dataset_frames=len(ds),labeled_hands=sum(len(l['cls']) for l in ds.labels),
                        loss=serial(loss),components=serial(components),
                        backward_finite=True,seconds=time.time()-started))
    del model,pretrained,ds,batch;torch.cuda.empty_cache()
(root/'provenance/yolo_smoke.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
