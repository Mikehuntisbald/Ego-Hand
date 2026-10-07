"""3D prototype checks; synthetic coarse errors are tests, not benchmark results."""
import json
from pathlib import Path
import torch
from pose_residual_dit import PoseResidualDiT
root=Path('/mnt/why/HOT3D');torch.manual_seed(20261002)
path=root/'export/annotations/train/P0001_9b6feab7/clip-001892.jsonl'
frame=json.loads(path.read_text().splitlines()[0])
gt=torch.tensor([h['xyz_camera_m'] for h in frame['hands']],device='cuda:0')
coarse=gt+.01*torch.randn_like(gt);conf=torch.full(gt.shape[:2],.7,device=gt.device)
rgb=torch.randn(gt.shape[0],16,64,device=gt.device) # shape-only feature stub
model=PoseResidualDiT(rgb_dim=64,width=96,depth=2,heads=4).to(gt.device)
assert torch.allclose(model.unpack(model.pack(gt)),gt,atol=1e-6)
result=model.loss(gt,coarse,conf,rgb)
result['loss'].backward()
assert torch.isfinite(result['loss'])
assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
perfect=model.loss(gt,gt,conf,rgb)
assert perfect['gate_target'].sum()==0,'Perfect initial pose must never label a nonzero correction beneficial'
identity=model.sample(coarse,conf,rgb,residual_strength=0)
assert torch.equal(identity['xyz_camera_m'],coarse)
sample=model.sample(coarse,conf,rgb,sampling_steps=4)
assert torch.isfinite(sample['xyz_camera_m']).all()
assert sample['gates'][:,model.wrist+1].sum()==0
report=dict(stage='untrained_prototype_smoke_only',finite_backward=True,pack_unpack_max_error_m=float((model.unpack(model.pack(gt))-gt).abs().max()),
            zero_strength_preserves_coarse_exactly=True,perfect_coarse_gate_target_zero=True,
            sampling_finite=True,parameters=sum(p.numel() for p in model.parameters()),
            note='Synthetic coarse error and stub RGB features; not a 3D pose accuracy evaluation')
(root/'provenance/dit_smoke.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
