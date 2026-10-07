"""Predeclared pose-supervision ablation; dev_select only, v16 stays default."""
import argparse, json, time
import numpy as np
import torch
from torch.nn import functional as F
from hand3d_v8_common import V7, save, metrics, score
from parameter_codec_v31 import OUT, observation_batch
from parameter_temporal_model_v31 import ParameterTemporalHand
from joint_mano_model_v29 import six_to_rotation


class PoseSupervisedHand(ParameterTemporalHand):
    def generate(self, *args, **kwargs):
        output = super().generate(*args, **kwargs)
        self.current_generated = output
        return output

    def objective(self, b, target, seed):
        loss, parts = super().objective(b, target, seed)
        state = self.current_generated['state'].flatten(2)
        teacher = target['state'].flatten(2)
        mask = target['mask'].float()
        def masked(values):
            return (values * mask).sum() / mask.sum().clamp_min(1)
        angles = masked((state[:, :, 9:29] - teacher[:, :, 9:29]).square().mean(-1))
        rotation = masked((six_to_rotation(state[:, :, 3:9]) - six_to_rotation(teacher[:, :, 3:9])).square().mean((-1, -2)))
        root = masked(F.smooth_l1_loss(state[:, :, :3], teacher[:, :, :3], reduction='none', beta=.1).mean(-1))
        # Extra pose supervision follows the fixed oracle diagnostic. Root
        # supervision also discourages compensating wrong pose with translation.
        loss = loss + 8 * angles + 4 * rotation + 2 * root
        parts.update(angle_mse=float(angles.detach()), rotation_mse=float(rotation.detach()), root_huber=float(root.detach()))
        return loss, parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4000)
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    torch.set_num_threads(4); torch.manual_seed(202610141)
    run = V7.parent / 'parameter_pose_v32'
    run.mkdir(exist_ok=True)
    assert not (run / 'done.json').exists()
    data = torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt', weights_only=False, mmap=True)
    data = {k:v.to(a.device) if torch.is_tensor(v) else v for k,v in data.items()}
    targets = {k:v.to(a.device) for k,v in torch.load(OUT/'refined_targets.pt', weights_only=False, mmap=True).items()}
    coarse = torch.load(OUT/'coarse_state_bank.pt', weights_only=False, mmap=True).to(a.device)
    right = torch.load(OUT/'predicted_right_bank.pt', weights_only=False).to(a.device)
    probabilities = torch.load(V7.parent/'side_data_v16/consensus/risk_dense/risk_probabilities.pt', weights_only=False)
    trainprob, devprob = probabilities['train_oof'].to(a.device), probabilities['joint'].to(a.device)
    bank = torch.load(V7.parent/'side_data_v16/consensus/native_bank.pt', weights_only=False, mmap=True).to(a.device)
    train = torch.tensor([i for i,r in enumerate(data['roles']) if r=='train'], device=a.device)
    dev = torch.tensor([i for i,r in enumerate(data['roles']) if r=='dev_select'], device=a.device)
    model = PoseSupervisedHand('regression', a.device).to(a.device)
    ck = torch.load(OUT/'regression/best.pt', weights_only=False, map_location=a.device)
    model.load_state_dict(ck['model'])
    config = dict(kind='regression', initial='v31 regression selected step2000', steps=a.steps, seed=202610141,
                  change='Additional8*normalized-angleMSE +4*SO3matrixMSE +2*rootHuber; unchanged RGB/data/decoder/protection',
                  reason='Posthoc parameter replacement diagnosed angle/global pose error; oracle never used at inference',
                  evaluation='Only dev_select selection; no dev_calibrate or fresh clips', default_changed=False)
    save(run/'config.json', config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.00005, weight_decay=.01)
    def get(ids, probability):
        b = observation_batch(data, ids, probability, coarse, right)
        b['rgb_native'] = bank[data['feature_ids'][ids]]
        return b
    @torch.inference_mode()
    def evaluate():
        model.eval(); predictions=[]; states=[]; sides=[]
        for start in range(0, len(dev), 8):
            ix=dev[start:start+8]
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p=model.predict_parameters(get(ix, devprob), seed=202610131+start)
            predictions.append(p['xyz_camera_m']);states.append(p['state']);sides.append(p['right'])
        xyz=torch.cat(predictions)
        result=metrics(xyz, data['original_base_for_evaluation'][dev], data['gt'][dev], data['valid'][dev])
        rank, feasible=score(result)
        return rank, result, dict(prediction=xyz.cpu(),state=torch.cat(states).cpu(),right=torch.cat(sides).cpu(),indices=dev.cpu())
    start=time.time();best=None;history=[]
    rank,m,p=evaluate();save(run/'matched_start.json',m)
    # Initial checkpoint is a legitimate selection option; prevent a worse
    # warm-start from silently replacing it.
    best=rank
    torch.save(dict(model=model.state_dict(),step=0,kind='regression',config=config),run/'best.pt')
    torch.save(p,run/'development_predictions.pt')
    for step in range(1,a.steps+1):
        model.train();ix=train[torch.randint(len(train),(8,),device=a.device)]
        b=get(ix,trainprob);target={k:v[ix] for k,v in targets.items()}
        lr=.00005*(.2+.8*.5*(1+np.cos(np.pi*step/a.steps)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.objective(b,target,202610141+step)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        optimizer.step()
        if step%250==0:
            rank,m,p=evaluate();entry=dict(step=step,loss=float(loss.detach()),parts=parts,metrics=m,rank=list(rank),seconds=time.time()-start)
            history.append(entry);save(run/'history.json',history)
            if rank<best:
                best=rank;torch.save(dict(model=model.state_dict(),step=step,kind='regression',config=config),run/'best.pt')
                torch.save(p,run/'development_predictions.pt')
            print(json.dumps(entry),flush=True)
    ck=torch.load(run/'best.pt',weights_only=False,map_location='cpu')
    save(run/'done.json',dict(complete=True,steps=a.steps,selected_step=ck['step'],seconds=time.time()-start,default_changed=False))
    print((run/'done.json').read_text(),flush=True)


if __name__=='__main__':main()
