"""Same auxiliary-attention capacity; compare original versus canonical RGB."""
import argparse, hashlib, sys
from pathlib import Path
import torch
from hand3d_v8_common import V7, save
import train_aligned_density_v13 as base_worker

CODE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['control','canonical'], required=True)
    ap.add_argument('--device', required=True)
    args = ap.parse_args()
    data = V7.parent / 'context_data_v17/control'
    run = V7.parent / 'dual_visual_v20' / args.variant
    run.mkdir(parents=True,exist_ok=True)
    extra_root = data if args.variant == 'control' else V7.parent/'canonical_rgb_v19'
    # Both variants receive the same source risk, XYZ, localization and time.
    extra_data = torch.load(extra_root/'dense_data.pt',weights_only=False,mmap=True)
    extra_bank = torch.load(extra_root/'native_bank.pt',weights_only=False,mmap=True).to(args.device)
    positions = extra_data['positions_bank'].to(args.device)
    rays = extra_data['rays_world'].to(args.device)
    original_make = base_worker.make
    def make(data, bank, ids, probability):
        b = original_make(data,bank,ids,probability)
        feature_ids = data['feature_ids'][ids]
        b['extra_rgb_native'] = extra_bank[feature_ids]
        b['extra_positions'] = positions[feature_ids]
        b['extra_rays'] = torch.einsum('btsc,bck->btsk',rays[feature_ids],data['rotation'][feature_ids[:,8]])
        return b
    base_worker.make = make
    source = (CODE/'train_recovery_v15.py').read_text()
    changes = {
        'from hand3d_trajectory_data_v9 import targets': "def targets(data,extra,ids):\n    return tuple(extra[k][ids] for k in ['gt','valid','gt_uv','uv_valid'])",
        'from recovery_model_v15 import RecoveryHand3D': 'from dual_visual_condition_v20 import DualVisualHand3D',
        "DATA/'trajectory_targets.pt'": "DATA/'trajectory_labels.pt'",
        "base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids]": "base=data['original_base_for_evaluation'][ids];gt=data['gt'][ids]",
        "base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];b=make(data,bank,dev,ep)": "base=data['original_base_for_evaluation'][dev];b=make(data,bank,dev,ep)",
        "model=RecoveryHand3D(policy,args.arm!='hard_conservative').to(args.device);model.load_state_dict(ck['model'])": "model=DualVisualHand3D(policy).to(args.device);mismatch=model.load_state_dict(ck['model'],strict=False);assert not mismatch.unexpected_keys and all(k.startswith(('extra_',)) for k in mismatch.missing_keys);model.initialize_extra()",
        "opt=torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=.04)": "groups=[dict(params=[v for k,v in model.named_parameters() if not k.startswith('extra_')],lr=5e-6,task_lr=5e-6),dict(params=[v for k,v in model.named_parameters() if k.startswith('extra_') and k!='extra_gain'],lr=5e-5,task_lr=5e-5),dict(params=[model.extra_gain],lr=5e-4,task_lr=5e-4)];opt=torch.optim.AdamW(groups,weight_decay=.04)",
        "group['lr']=5e-6*": "group['lr']=group['task_lr']*",
        "save(run/'gradient_check.json',dict(passed=True,**norms))": "norms['extra_gain']=float(model.extra_gain.grad.norm());assert norms['extra_gain']>0;save(run/'gradient_check.json',dict(passed=True,**norms))",
        "if step%100==0:print": "if step==100:\n            save(run/'extra_branch_update.json',dict(gain=model.extra_gain.detach().cpu().tolist(),projection_changed=float((model.extra_rgb_project.weight-ck['model']['rgb_project.weight']).norm()),attention_changed=float((model.extra_attention[0].in_proj_weight-ck['model']['blocks.0.cross_attention.in_proj_weight']).norm())))\n        if step%100==0:print",
    }
    generated = source
    for before, after in changes.items():
        assert generated.count(before)==1, before
        generated = generated.replace(before,after)
    snapshot = run/'trainer_snapshot.py'
    snapshot.write_text(generated)
    save(run/'source_provenance.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),
        generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),replacements=changes,
        variant=args.variant,scope='Same900updates/seed/initializer/labels/risk/XYZ/times/originalRGB/localization/capacity. Auxiliary semantic memory uses original vs canonical predicted-side RGB and matching physical geometry. Zero initial gain. No new evaluation batch.'))
    ns = dict(__name__='dual_visual_worker_v20',__file__=str(snapshot))
    exec(compile(generated,str(snapshot),'exec'),ns)
    ns.update(DATA=data,RUN=run,INITIAL=V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt')
    sys.argv = ['train','--arm','uniform_adaptive','--device',args.device,'--steps','900']
    ns['main']()

if __name__=='__main__':
    main()
