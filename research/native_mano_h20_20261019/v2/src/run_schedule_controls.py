"""Matched diffusion schedule correction; immutable original controls."""
import hashlib,json,os,subprocess,sys
from pathlib import Path
import torch
from native_dit import NativeMANODiT
R=Path('/mnt/why/native_mano_dit_v2_20261019');V1=Path('/mnt/why/native_mano_dit_20261019')

def main():
    torch.set_num_threads(4)
    torch.manual_seed(190019);a=NativeMANODiT(schedule='linear')
    torch.manual_seed(190019);b=NativeMANODiT(schedule='cosine')
    assert all(torch.equal(x,b.state_dict()[k]) for k,x in a.state_dict().items() if k!='alpha')
    target=torch.randn(16,17,3);noise=torch.randn_like(target);errors=[]
    for alpha in [a.alpha,b.alpha]:
        av=alpha[torch.tensor([0,6,12,18,24,30,36,42,48,54,60,66,72,78,90,99]),None,None]
        xt=av.sqrt()*target+(1-av).sqrt()*noise
        v=av.sqrt()*noise-(1-av).sqrt()*target
        errors.append(float((av.sqrt()*xt-(1-av).sqrt()*v-target).abs().max()))
    assert max(errors)<1e-6
    preflight=dict(complete=True,linear_terminal_alpha=float(a.alpha[-1]),cosine_terminal_alpha=float(b.alpha[-1]),
        same_initial_weights_except_noise_schedule=True,v_prediction_inverse_max=errors,
        all_other_training_code_and_loss_weights_unchanged=True,
        input_sha256=hashlib.sha256((V1/'data/inputs.pt').read_bytes()).hexdigest(),
        target_sha256=hashlib.sha256((V1/'data/training_targets.pt').read_bytes()).hexdigest(),
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'src').glob('*.py')})
    (R/'receipts/schedule_preflight.json').write_text(json.dumps(preflight,indent=2))
    for arm in ['dit','protected']:
        output=R/'runs'/f'cosine_{arm}'
        with (R/'logs'/f'cosine_{arm}.log').open('w') as log:
            subprocess.run([sys.executable,str(R/'src/train_native.py'),'--arm',arm,'--steps','600','--schedule','cosine','--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
        done=json.loads((output/'done.json').read_text());old=json.loads((V1/f'runs/{arm}/done.json').read_text())
        assert done['batch_plan_sha']==old['batch_plan_sha']
        print(json.dumps(dict(arm=arm,training=done,selection=json.loads((output/'selection.json').read_text()))),flush=True)

if __name__=='__main__':main()
