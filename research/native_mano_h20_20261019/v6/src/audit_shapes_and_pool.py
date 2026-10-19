"""Training-only audit of shape/pose target compatibility and data capacity."""
import collections,hashlib,json,sys
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_mano_shape_v6_20261019');V5=Path('/mnt/why/native_mano_supervision_v5_20261019')
sys.path.insert(0,str(V5/'src'));sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src')
from native_data_v5 import Data
from native_mano import NativeMANO
def stats(x):
    x=x.detach().cpu().numpy();return dict(n=x.size,mean=float(x.mean()),P95=float(np.quantile(x,.95)),max=float(x.max()))
def main():
    torch.set_num_threads(4);R.mkdir(exist_ok=True);(R/'receipts').mkdir(exist_ok=True)
    old=Path('/mnt/why/HOT3D/experiments/online_rgb_3d_v47/inputs.pt');d=torch.load(old,weights_only=False,mmap=True,map_location='cpu')
    groups=collections.Counter((role,subject,row['sequence'],int(row['clip'])) for role,subject,row in zip(d['roles'],d['subjects'],d['rows']))
    counts=dict(roles=dict(collections.Counter(d['roles'])),subjects=dict(collections.Counter(d['subjects'])),clips=[dict(role=k[0],subject=k[1],sequence=k[2],clip=k[3],windows=v) for k,v in sorted(groups.items())],observations=len(d['world'])-1)
    data=Data();codec=NativeMANO();ids=data.train;b=data.batch(ids);t=data.target(ids)
    with torch.no_grad():
        exact_pose=codec.decode(t['gt'][:,0],t['root_rotation'],t['local_rotation'],b['betas'],b['right'])
        compatible=exact_pose-exact_pose[:,:1];actual=t['gt']-t['gt'][:,:1]
        shaped_error=(compatible-actual).norm(dim=-1)[:,1:]*1000
        bounded=codec.apply_residual(b['state'],b['betas'],b['right'],t['delta'])['joints'];bounded=bounded-bounded[:,:1]
        native=codec.decode(t['gt'][:,0],t['root_rotation'],t['local_rotation'],t['beta_GT'],b['right']);parity=float((native-t['gt']).abs().max());assert parity<2e-6
        # A perfect GT rotation has zero compatible FK target loss while retaining predicted beta as input.
        compatible_loss=float(((compatible-compatible).square()).mean())
    report=dict(complete=True,training_right_windows=len(ids),data_pool=counts,training_shape_only_native_relative20_mm=stats(shaped_error),
        bounded_rotation_plus_shape_native_relative20_mm=stats((bounded-actual).norm(dim=-1)[:,1:]*1000),bounded_vs_compatible_pose_target_mm=stats((bounded-compatible).norm(dim=-1)[:,1:]*1000),
        true_pose_under_predicted_shape_compatible_loss=compatible_loss,true_pose_and_GT_shape_FK_parity_max_m=parity,
        hypothesis='GT native FK at fixed predicted beta removes unreachable shape displacement from pose loss; real native GT at actual beta remains evaluation and accurate-point protection target',
        no_validation_labels_opened=True,production_admitted=False)
    (R/'receipts/shape_and_pool_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
