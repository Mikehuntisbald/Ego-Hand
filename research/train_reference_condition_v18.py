"""Same input/labels/seed/updates, reference enabled or disabled."""
import argparse,hashlib,sys
from pathlib import Path
from hand3d_v8_common import V7,save
CODE=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('--variant',choices=['control','reference'],required=True);p.add_argument('--device',required=True);a=p.parse_args()
DATA=V7.parent/'context_data_v17/control';RUN=V7.parent/'reference_condition_v18'/a.variant;RUN.mkdir(parents=True,exist_ok=True)
source=(CODE/'train_recovery_v15.py').read_text();changes={
    'from hand3d_trajectory_data_v9 import targets':"def targets(data,extra,ids):\n    return tuple(extra[k][ids] for k in ['gt','valid','gt_uv','uv_valid'])",
    'from recovery_model_v15 import RecoveryHand3D':'from reference_condition_v18 import ReferenceConditionHand3D',
    "DATA/'trajectory_targets.pt'":"DATA/'trajectory_labels.pt'",
    "base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids]":"base=data['original_base_for_evaluation'][ids];gt=data['gt'][ids]",
    "base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];b=make(data,bank,dev,ep)":"base=data['original_base_for_evaluation'][dev];b=make(data,bank,dev,ep)",
    "model=RecoveryHand3D(policy,args.arm!='hard_conservative').to(args.device);model.load_state_dict(ck['model'])":"model=ReferenceConditionHand3D(policy,REFERENCE_ENABLED).to(args.device);mismatch=model.load_state_dict(ck['model'],strict=False);assert all(k.startswith('reference_project.') for k in mismatch.missing_keys) and not mismatch.unexpected_keys"}
generated=source
for before,after in changes.items():assert generated.count(before)==1;generated=generated.replace(before,after)
snapshot=RUN/'trainer_snapshot.py';snapshot.write_text(generated);save(RUN/'source_provenance.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),replacements=changes,variant=a.variant,scope='Zero initialized additional reference condition; no base replacement; fullRGB and identical labels/updates retained'))
ns=dict(__name__='reference_condition_worker_v18',__file__=str(snapshot));exec(compile(generated,str(snapshot),'exec'),ns);ns['DATA']=DATA;ns['RUN']=RUN;ns['INITIAL']=V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt';ns['REFERENCE_ENABLED']=a.variant=='reference';sys.argv=['train','--arm','uniform_adaptive','--device',a.device,'--steps','900'];ns['main']()
