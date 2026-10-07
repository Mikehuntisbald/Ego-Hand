import json,torch
from train_coarse_pose import PoseDataset,RUN
from coarse_pose3d import CoarsePose3D,coarse_loss
from torch.utils.data import DataLoader
torch.set_num_threads(4)
ds=PoseDataset(RUN/'warmup_manifest.jsonl',roles=['coarse']);batch=next(iter(DataLoader(ds,batch_size=4)))
model=CoarsePose3D().to('cuda:1').train();batch={k:v.to('cuda:1') if torch.is_tensor(v) else v for k,v in batch.items()}
pred=model(batch['image'],batch['geometry']);loss,parts=coarse_loss(pred,batch['gt'],batch['uv'],batch['valid'])
loss.backward();assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
assert pred['xyz'].shape==(4,20,3) and pred['rgb_tokens'].shape==(4,64,128)
assert torch.allclose(pred['relative'][:,5],torch.zeros(4,3,device='cuda:1'))
print(json.dumps(dict(loss=float(loss.detach()),finite_backward=True,geometry=tuple(batch['geometry'].shape),xyz_shape=tuple(pred['xyz'].shape))),flush=True)
