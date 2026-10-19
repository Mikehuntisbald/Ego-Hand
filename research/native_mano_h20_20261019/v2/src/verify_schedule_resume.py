"""Exact optimizer/RNG roundtrip, not a resumed-training quality claim."""
import json,random
from pathlib import Path
import numpy as np,torch
from native_dit import NativeMANODiT
R=Path('/mnt/why/native_mano_dit_v2_20261019')

def main():
    reports={}
    for arm in ['dit','protected']:
        ck=torch.load(R/f'runs/cosine_{arm}/last.pt',weights_only=False,map_location='cuda')
        model=NativeMANODiT(**ck['config']).cuda();model.load_state_dict(ck['model'])
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4);opt.load_state_dict(ck['optimizer'])
        assert all(torch.equal(v,ck['model'][k]) for k,v in model.state_dict().items())
        assert opt.state_dict()['param_groups']==ck['optimizer']['param_groups']
        states=opt.state_dict()['state'];assert len(states)==len(ck['optimizer']['state'])
        for key,parts in states.items():
            for field,value in parts.items():
                saved=ck['optimizer']['state'][key][field]
                if torch.is_tensor(value):assert torch.equal(value.cpu(),saved.cpu())
                else:assert value==saved
        def draw():
            torch.set_rng_state(ck['rng_cpu'].cpu());torch.cuda.set_rng_state(ck['rng_cuda'].cpu())
            g=torch.Generator(device='cuda');g.set_state(ck['noise_rng'].cpu())
            random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng'])
            return [torch.randn(16),torch.randn(16,device='cuda').cpu(),torch.randn(16,device='cuda',generator=g).cpu(),np.random.rand(16),random.random()]
        a,b=draw(),draw();assert all(torch.equal(x,y) if torch.is_tensor(x) else np.array_equal(x,y) for x,y in zip(a,b))
        reports[arm]=dict(step=ck['step'],schedule=ck['config']['schedule'],weights_exact=True,optimizer_exact=True,RNG_next_draw_exact=True,
                          actual_interrupted_training_continuation_test=False)
    (R/'receipts/cosine_resume_roundtrip.json').write_text(json.dumps(dict(complete=True,arms=reports),indent=2));print(json.dumps(reports))

if __name__=='__main__':main()
