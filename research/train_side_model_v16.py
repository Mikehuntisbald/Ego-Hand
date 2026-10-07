"""Reuse verified sampler training; compare the full pipeline against old XYZ."""
import argparse,hashlib,json,sys
from pathlib import Path
from hand3d_v8_common import V7,save
CODE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['control','consensus'],required=True);p.add_argument('--device',required=True);a=p.parse_args()
DATA=V7.parent/'side_data_v16'/a.variant;RUN=V7.parent/'side_native_v16'/a.variant
assert json.loads((DATA.parent/'upstream_results.json').read_text())['approved_for_3d_training']
RUN.mkdir(parents=True,exist_ok=True);source=(CODE/'train_recovery_v15.py').read_text()
changes={
    "base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids]":"base=data['original_base_for_evaluation'][ids];gt=data['gt'][ids]",
    "base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];b=make(data,bank,dev,ep)":"base=data['original_base_for_evaluation'][dev];b=make(data,bank,dev,ep)"}
generated=source
for before,after in changes.items():assert generated.count(before)==1;generated=generated.replace(before,after)
snapshot=RUN/'trainer_snapshot.py';snapshot.write_text(generated)
save(RUN/'source_provenance.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),replacements=changes,variant=a.variant,role='Matchedsameinitialization/seed/steps; validationdifficultyandharmcomparetoidenticaloriginalWiLoRXYZ; noGTininference'))
ns=dict(__name__='side_model_v16_worker',__file__=str(snapshot));exec(compile(generated,str(snapshot),'exec'),ns)
ns['DATA']=DATA;ns['RUN']=RUN
sys.argv=['train_recovery_v15.py','--arm','uniform_adaptive','--device',a.device,'--steps','900'];ns['main']()
