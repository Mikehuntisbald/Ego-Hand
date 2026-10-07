"""Matched current-risk broadcast versus perframe perjoint reliability."""
import argparse,hashlib,sys
from pathlib import Path
import torch
from hand3d_v8_common import V7,save
import train_aligned_density_v13 as base_worker

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',choices=['control','temporal'],required=True)
    ap.add_argument('--device',required=True)
    a=ap.parse_args()
    code=Path(__file__).resolve().parent
    data=V7.parent/'context_data_v17/control'
    run=V7.parent/'temporal_condition_v22'/a.variant
    run.mkdir(parents=True,exist_ok=True)
    temporal=torch.load(V7.parent/'temporal_reliability_v22/risk_probabilities.pt',weights_only=False)['train_oof'].to(a.device)
    old_make=base_worker.make
    def make(data,bank,ids,prob):
        b=old_make(data,bank,ids,prob)
        b['temporal_error_risk']=(temporal[ids] if a.variant=='temporal' else
            torch.stack([b['risk_camera'],b['risk_relative']],-1)[:,None].expand(-1,17,-1,-1))
        return b
    base_worker.make=make
    source=(code/'train_recovery_v15.py').read_text()
    changes={
        'from hand3d_trajectory_data_v9 import targets':"def targets(data,extra,ids):\n    return tuple(extra[k][ids] for k in ['gt','valid','gt_uv','uv_valid'])",
        'from recovery_model_v15 import RecoveryHand3D':'from temporal_risk_condition_v22 import TemporalRiskHand3D',
        "DATA/'trajectory_targets.pt'":"DATA/'trajectory_labels.pt'",
        "base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids]":"base=data['original_base_for_evaluation'][ids];gt=data['gt'][ids]",
        "base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];b=make(data,bank,dev,ep)":"base=data['original_base_for_evaluation'][dev];b=make(data,bank,dev,ep)",
        "model=RecoveryHand3D(policy,args.arm!='hard_conservative').to(args.device);model.load_state_dict(ck['model'])":"model=TemporalRiskHand3D(policy).to(args.device);mismatch=model.load_state_dict(ck['model'],strict=False);assert not mismatch.unexpected_keys and all(k.startswith('temporal_risk_project.') for k in mismatch.missing_keys)",
        "opt=torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=.04)":"groups=[dict(params=[v for k,v in model.named_parameters() if not k.startswith('temporal_risk_project.')],lr=5e-6,task_lr=5e-6),dict(params=model.temporal_risk_project.parameters(),lr=5e-5,task_lr=5e-5)];opt=torch.optim.AdamW(groups,weight_decay=.04)",
        "group['lr']=5e-6*":"group['lr']=group['task_lr']*",
        "save(run/'gradient_check.json',dict(passed=True,**norms))":"norms['temporal_risk_condition']=float(model.temporal_risk_project[-1].weight.grad.norm());assert norms['temporal_risk_condition']>0;save(run/'gradient_check.json',dict(passed=True,**norms))",
    }
    generated=source
    for before,after in changes.items():
        assert generated.count(before)==1,before
        generated=generated.replace(before,after)
    snapshot=run/'trainer_snapshot.py';snapshot.write_text(generated)
    save(run/'source_provenance.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),replacements=changes,variant=a.variant,
        scope='Same900updates/seed/initializer/RGB/XYZ/slots/labels/currentrisk/policy/addedcapacity. Historical token conditions use originalcurrent risk broadcast or perframe subject-excluded error risk. Zero initial addedembedding. Current gates untouched; no test/retainedfailures.'))
    ns=dict(__name__='temporal_risk_worker_v22',__file__=str(snapshot));exec(compile(generated,str(snapshot),'exec'),ns)
    ns.update(DATA=data,RUN=run,INITIAL=V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt')
    sys.argv=['train','--arm','uniform_adaptive','--device',a.device,'--steps','900']
    ns['main']()

if __name__=='__main__':main()
