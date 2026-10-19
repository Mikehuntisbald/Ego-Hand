"""Native FK/export/gradient checks, separate from any quality claim."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
from native_mano import NativeMANO,exp_so3
ROOT=Path('/mnt/why/native_mano_dit_20261019')
torch.set_num_threads(4);decoder=NativeMANO()
source=np.load('/mnt/why/public_headhand_h20_20261019/outputs/wilor_clip2102/observations.npz')
state=torch.tensor(source['state'][:8],device='cuda');beta=torch.tensor(source['betas'][:8],device='cuda');right=torch.tensor(source['right'][:8],device='cuda')
delta=torch.zeros(8,17,3,device='cuda',requires_grad=True)
result=decoder.apply_residual(state,beta,right,delta);native=decoder.export_native(result);redecoded=decoder.decode_exported(native)
error=float((result['joints']-redecoded).abs().max());assert error<1e-6,error
parity=float((result['joints']-torch.tensor(source['xyz_camera'][:8],device='cuda')).abs().max());assert parity<1e-6,parity
with torch.autocast('cuda',dtype=torch.bfloat16):
    mixed=decoder.apply_residual(state,beta,right,delta)
    mixed_export=decoder.export_native(mixed);mixed_back=decoder.decode_exported(mixed_export)
mixed_error=float((mixed_back-result['joints']).abs().max());assert mixed_error<1e-6,mixed_error
loss=(result['joints']*torch.arange(21,device='cuda')[None,:,None]).square().mean();loss.backward()
assert torch.isfinite(delta.grad).all() and delta.grad[:,2:].abs().max()>0
assert torch.equal(result['wrist'],state[:,:3]);assert torch.equal(result['betas'],beta)
rot=torch.cat([native['global_orient'].reshape(-1,3,3),native['hand_pose'].reshape(-1,3,3)])
orth=float((rot.transpose(-1,-2)@rot-torch.eye(3,device='cuda')).abs().max());det=float(torch.linalg.det(rot).sub(1).abs().max());assert orth<1e-5 and det<1e-5
report=dict(complete=True,native_parameter_redecode_max_m=error,original_FK_max_m=parity,
            SO3_orthogonality_max=orth,SO3_determinant_error_max=det,local_gradient_max=float(delta.grad[:,2:].abs().max()),
            wrist_unchanged=True,betas_unchanged=True,outer_BF16_geometry_max_m=mixed_error,GT_used=False,scope='geometry and gradient preflight only; no accuracy claim')
(ROOT/'receipts/native_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
