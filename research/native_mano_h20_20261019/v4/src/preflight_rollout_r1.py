"""Sampler parity and isolated sampler-to-RGB/temporal/FK gradient check."""
import hashlib,json
from pathlib import Path
import torch
from native_rollout import NativeRolloutDiT
from native_dit import NativeMANODiT
from native_data import Data
from native_mano import NativeMANO,common
R=Path('/mnt/why/native_mano_rollout_v4_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019')

def main():
    torch.set_num_threads(4);data=Data();codec=NativeMANO();b=data.batch(data.train[:3]);t=data.target(data.train[:3]);path=V2/'runs/cosine_dit/best_development.pt';ck=torch.load(path,weights_only=False,map_location='cuda')
    torch.manual_seed(190019);plain=NativeMANODiT(schedule='cosine').cuda()
    torch.manual_seed(190019);model=NativeRolloutDiT(schedule='cosine').cuda()
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in plain.state_dict().items())
    model.load_state_dict(ck['model']);model.eval();g=torch.Generator(device='cuda').manual_seed(190019)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        inference=model.sample(b,draws=4,steps=10,seed=190019)
    # End the no-grad autocast context before building a gradient graph.
    # Otherwise its cached cast weights can be detached from Parameters.
    with torch.autocast('cuda',dtype=torch.bfloat16):
        actual=model.sample_train(b,g,draws=4,steps=10)
    assert torch.equal(inference,actual),float((inference-actual).abs().max())
    result=codec.apply_residual(b['state'],b['betas'],b['right'],actual.mean(0).float());p=result['joints'][:,common.MAPPING];gt=t['gt'];relative=(p-p[:,5:6])-(gt-gt[:,5:6]);loss=relative.square().mean();loss.backward()
    gradients={prefix:max(float(x.grad.abs().max()) for n,x in model.named_parameters() if n.startswith(prefix) and x.grad is not None) for prefix in ['rgb_stem','spatial','temporal','head']}
    assert all(v>0 for v in gradients.values()),gradients
    assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
    # Geometry must stay native MANO even with a gradient-enabled sampler.
    native=codec.export_native(result);error=float((codec.decode_exported(native)-result['joints']).abs().max());assert error<1e-6
    report=dict(complete=True,training_sampler_equals_deployment_samples_exact=True,no_grad_autocast_cache_isolated=True,sampler_only_gradient_max=gradients,all_gradients_finite=True,native_redecode_max_m=error,train_rows_only=True,no_validation_hand_labels_read=True,
        objective_change='same v/FK/parameter/velocity/protection coefficients; replace one-step FK source by actual pure-noise 4-draw 10-step mean',same_updates_not_equal_FLOPs=True,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'src').glob('*.py')})
    (R/'receipts/preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
