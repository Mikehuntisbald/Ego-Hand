"""Matched update budget, original versus expanded real current-frame targets."""
import argparse,hashlib,sys
from pathlib import Path
from hand3d_v8_common import V7,save

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--variant',choices=['control','expanded'],required=True);ap.add_argument('--stage',choices=['risk','model'],required=True);ap.add_argument('--device',required=True);a=ap.parse_args()
    code=Path(__file__).resolve().parent;data=V7.parent/('context_data_v17/control' if a.variant=='control' else 'dense_centers_v26');run=V7.parent/'dense_training_v26'/a.variant;run.mkdir(parents=True,exist_ok=True)
    if a.stage=='risk':
        assert a.variant=='expanded'
        import train_density_risk_v13 as worker
        worker.ROOT=data;sys.argv=['risk','--arm','dense','--device',a.device];worker.main();return
    source=(code/'train_recovery_v15.py').read_text();changes={
        'from hand3d_trajectory_data_v9 import targets':"def targets(data,extra,ids):\n    return tuple(extra[k][ids] for k in ['gt','valid','gt_uv','uv_valid'])",
        "DATA/'trajectory_targets.pt'":"DATA/'trajectory_labels.pt'",
        "base=data['xyz_camera_bank'][data['feature_ids'][ids,8]];gt=data['gt'][ids]":"base=data['original_base_for_evaluation'][ids];gt=data['gt'][ids]",
        "base=data['xyz_camera_bank'][data['feature_ids'][dev,8]];b=make(data,bank,dev,ep)":"base=data['original_base_for_evaluation'][dev];b=make(data,bank,dev,ep)",
    }
    generated=source
    for before,after in changes.items():assert generated.count(before)==1,before;generated=generated.replace(before,after)
    snapshot=run/'trainer_snapshot.py';snapshot.write_text(generated)
    save(run/'source_provenance.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),replacements=changes,variant=a.variant,
        scope='Same6000updates/seed/batch4/initializer/architecture/policy/selection. Trainingcenters2526vs17291; expandedrisks refitted subject-excluded. OriginalRGB/XYZ banks and all352devselect/539devcalibrate inputs/labels unchanged. Denseframes already used as context, not newimages/subjects. No test/oldfailures used.'))
    ns=dict(__name__='dense_centers_v26_worker',__file__=str(snapshot));exec(compile(generated,str(snapshot),'exec'),ns)
    ns.update(DATA=data,RUN=run,INITIAL=V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt')
    sys.argv=['train','--arm','uniform_adaptive','--device',a.device,'--steps','6000'];ns['main']()

if __name__=='__main__':main()
