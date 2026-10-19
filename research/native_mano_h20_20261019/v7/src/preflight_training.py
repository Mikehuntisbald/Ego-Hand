"""Full-bank native sampler path, GT isolation, and old-small data comparison."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_mano_pool_v7_20261019');V5=Path('/mnt/why/native_mano_supervision_v5_20261019')
sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(V5/'src'));sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src');sys.path.insert(0,'/mnt/why/native_mano_rollout_v4_20261019/src')
from native_data_v7 import Data
from native_data_v5 import Data as OldData
from native_rollout import NativeRolloutDiT
from native_mano import NativeMANO
def main():
    torch.set_num_threads(4);data=Data(data_arm='full');old=OldData();protocol=json.loads((R/'data/selection.json').read_text());sources=set(protocol['small_source_indices']);small=[i for i in data.train if data.rows[i]['source_window_index'] in sources];assert len(small)==155
    oldmap={r['source_window_index']:i for i,r in enumerate(old.rows)};comparison={};GTmax=0.
    for begin in range(0,len(small),8):
        ids=small[begin:begin+8];previous=[oldmap[data.rows[i]['source_window_index']] for i in ids];current=data.batch(ids);ref=old.batch(previous)
        for k,v in current.items():
            if k=='indices':continue
            diff=float((v.float()-ref[k].float()).abs().max());comparison[k]=max(comparison.get(k,0),diff)
        target=data.target(ids);refGT=old.target(previous);GTmax=max(GTmax,float((target['gt']-refGT['gt']).abs().max()))
    assert GTmax<2e-6,'Small control native GT semantics changed'
    ck=torch.load(V5/'runs/native_dit/best_development.pt',map_location='cuda',weights_only=False);model=NativeRolloutDiT(**ck['config']).cuda().eval();model.load_state_dict(ck['model']);codec=NativeMANO()
    ids=data.train[:3];b=data.batch(ids);target=data.target(ids)
    with torch.autocast('cuda',dtype=torch.bfloat16):samples=model.sample(b,draws=4,steps=10,seed=190019)
    saved={k:v.clone() for k,v in data.labels.items() if v.is_floating_point()}
    for k in saved:data.labels[k].fill_(float('nan'))
    poisoned=data.batch(ids);assert all(torch.equal(v,poisoned[k]) for k,v in b.items())
    with torch.autocast('cuda',dtype=torch.bfloat16):resampled=model.sample(poisoned,draws=4,steps=10,seed=190019)
    assert torch.equal(samples,resampled)
    for k,v in saved.items():data.labels[k].copy_(v)
    generator=torch.Generator(device='cuda').manual_seed(190019)
    with torch.autocast('cuda',dtype=torch.bfloat16):differentiable=model.sample_train(b,generator,draws=4,steps=10)
    assert torch.equal(samples,differentiable)
    value=codec.apply_residual(b['state'],b['betas'],b['right'],differentiable.mean(0));xyz=value['joints'];gt=target['gt'];loss=((xyz-xyz[:,:1])-(gt-gt[:,:1])).square()[:,1:].mean();loss.backward()
    gradients={prefix:max(float(p.grad.abs().max()) for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None) for prefix in ['rgb_stem','spatial','temporal','head']};assert all(v>0 for v in gradients.values())
    parity=float((codec.decode_exported(codec.export_native(value))-xyz).detach().abs().max());assert parity<1e-6
    report=dict(complete=True,train_full_right=1330,train_small_right=155,dev_select_right=25,old_small_input_max_abs=comparison,old_small_GT_max_m=GTmax,
        current_small_and_full_share_identical_cache=True,GT_label_poison_unchanged=True,deployment_and_training_sampler_equal=True,sampler_gradient_max=gradients,native_export_redecode_max_m=parity,
        scope='Both new arms use same full cache. Old v5 input differences listed explicitly; old small windows kept by source IDs. Image backbone frozen; preflight validates actual spatial-feature/temporal/native FK path, not online backbone training.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (R/'receipts/training_preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
