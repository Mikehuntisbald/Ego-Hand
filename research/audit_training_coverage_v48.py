"""Audit loss-mask availability on training roles only; no RGB/inference changes."""
import hashlib,json,time
from pathlib import Path
import torch
ROOT=Path('/mnt/why/HOT3D/experiments')
SOURCE=ROOT/'online_rgb_3d_v47';RUN=ROOT/'online_rgb_iterative_v48'
torch.set_num_threads(4)
data=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True)
target=torch.load(SOURCE/'targets.pt',weights_only=False,mmap=True)
teacher=torch.load(RUN/'teacher_centers.pt',weights_only=False)
train=torch.tensor([i for i,r in enumerate(data['roles']) if r=='train'])
assert len(train)==2526 and all(data['roles'][int(i)]=='train' for i in train)
gt=target['gt'][train,8];valid=target['valid'][train,8].clone();valid[:,5]=False
raw=data['xyz_camera_bank'][data['feature_ids'][train,8]].float();teach=teacher[train].float()
rel=lambda x:x-x[:,5:6]
mask=lambda x,relative=False:valid&(((rel(x)-rel(gt)).norm(dim=-1) if relative else (x-gt).norm(dim=-1))<=.01)
original_camera=mask(raw);teacher_camera=mask(teach)
original_relative=mask(raw,True);teacher_relative=mask(teach,True)
groups={'original_camera':original_camera,'teacher_camera':teacher_camera,
    'union_camera':original_camera|teacher_camera,'original_relative':original_relative,
    'teacher_relative':teacher_relative,'union_relative':original_relative|teacher_relative}
plan=torch.load(RUN/'protected/batch_plan.pt',weights_only=False)
assert tuple(plan.shape)==(1800,2,2)
coverage={}
for name,m in groups.items():
    counts=m.sum(-1);available=counts>0;batch_counts=counts[plan]
    micro=batch_counts.sum(-1);effective=micro.sum(-1)
    coverage[name]=dict(protected_points=int(m.sum()),windows_with_any=int(available.sum()),
        windows_with_at_least3=int((counts>=3).sum()),windows_with_none=int((~available).sum()),
        fraction_windows_with_any=float(available.float().mean()),mean_protected_points_per_window=float(counts.float().mean()),
        microbatches_without_mask_points=int((micro==0).sum()),total_microbatches=int(micro.numel()),
        steps_without_mask_points=int((effective==0).sum()),total_steps=int(effective.numel()))
e=(rel(raw)-rel(gt)).norm(dim=-1)*1000;mean=(e*valid).sum(-1)/valid.sum(-1).clamp_min(1);hard=mean>40
result=dict(complete=True,exported_at=time.time(),training_windows=len(train),uses_training_roles_only=True,
    GT_only_supervised_sampling_and_loss_audit=True,GT_not_inference_input=True,
    threshold_good_mm=10,threshold_hard_mean_relative_mm=40,coverage=coverage,
    high_error_windows=int(hard.sum()),high_error_with_original_camera_good=int((hard&original_camera.any(-1)).sum()),
    high_error_with_union_camera_good=int((hard&(original_camera|teacher_camera).any(-1)).sum()),
    batch_plan_sha256=hashlib.sha256(plan.numpy().tobytes()).hexdigest(),
    visibility_annotation_available=False,
    interpretation='Mask availability only, not active hinge gradients. High-error groups are not occlusion truth. Dev/replay results not used.',
    training_configuration_changed=False,default_changed=False)
path=RUN/'training_coverage_audit.json';path.write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
