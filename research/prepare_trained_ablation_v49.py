"""Build a separate frozen derivative of the resumable v48 trainer/inferencer."""
import hashlib,json
from pathlib import Path
CODE=Path(__file__).parent
RUN=Path('/mnt/why/HOT3D/experiments/trained_module_ablation_v49_20261007')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    RUN.mkdir(parents=True,exist_ok=True);snap=RUN/'code_snapshot';snap.mkdir(exist_ok=True)
    def replace(text,old,new):
        assert text.count(old)==1,(old,text.count(old));return text.replace(old,new)
    t=(CODE/'train_online_parameter_v48.py').read_text()
    replacements=[
    ("from protected_parameter_model_v48 import ProtectedParameterHand","from module_training_model_v49 import ModuleAblationHand,transform_input,ARMS"),
    ("RUN=V7.parent/'online_rgb_iterative_v48'","RUN=V7.parent/'trained_module_ablation_v49_20261007'\nTEACHER=V7.parent/'online_rgb_iterative_v48'"),
    ("choices=['frozen','joint','protected']","choices=ARMS"),
    ("ap.add_argument('--resume',action='store_true');ap.add_argument('--preflight-only',action='store_true')","ap.add_argument('--resume',action='store_true');ap.add_argument('--preflight-only',action='store_true');ap.add_argument('--stop-at',type=int)"),
    ("(RUN/'teacher_ready.json')","(TEACHER/'teacher_ready.json')"),
    ("source_mode='frozen' if args.arm=='frozen' else 'joint';joint=args.arm!='frozen'","source_mode='joint';joint=args.arm not in ['frozen_same_start','no_rgb']"),
    ("target['teacher_xyz']=torch.load(RUN/'teacher_centers.pt'","target['teacher_xyz']=torch.load(TEACHER/'teacher_centers.pt'"),
    ("if args.arm=='protected':\n        replacement=ProtectedParameterHand('dit',args.device,teacher_weight=4.).to(args.device)\n        replacement.load_state_dict(model.state_dict());model=replacement", "replacement=ModuleAblationHand('dit',args.device,arm=args.arm).to(args.device)\n    replacement.load_state_dict(model.state_dict());model=replacement"),
    ("digest(RUN/'teacher_centers.pt')","digest(TEACHER/'teacher_centers.pt')"),
    ("teacher_guard_weight=4. if args.arm=='protected' else 0.","teacher_guard_weight=model.teacher_weight"),
    ("if (folder/'config.json').exists():","config.pop('stop_at',None)\n    config.update(arm=args.arm,shared_warmstart_all_arms=True,initial_visual_tail_saved_even_if_frozen=True,\n        training_ablation=True,default_changed=False,comparison='Equal warmstart/data order/1800 continuation steps/effectivebatch4; GPU compute differs and is reported. Not from-scratch ablation.')\n    if (folder/'config.json').exists():"),
    ("return visual.replace(observation_batch(data,ids,risk,coarse,right,preserve_fitted=True),data['feature_ids'][ids],pixels)","bb=observation_batch(data,ids,risk,coarse,right,preserve_fitted=True)\n        if args.arm=='no_rgb':bb['rgb_native']=torch.zeros(len(ids),17,192,1280,device=args.device)\n        else:bb=visual.replace(bb,data['feature_ids'][ids],pixels)\n        return transform_input(bb,args.arm)"),
    ("teacher_dev_path=RUN/f'teacher_dev_batch{args.batch}.pt'","teacher_dev_path=TEACHER/f'teacher_dev_batch{args.batch}.pt'\n    assert teacher_dev_path.exists(),'Use frozen teacher evaluation, never evaluate teacher with ablated student input'"),
    ("visual_tail=visual.tail_state() if joint else None","visual_tail=visual.tail_state()"),
    ("for step in range(first_step+1,args.steps+1):","for step in range(first_step+1,min(args.steps,args.stop_at or args.steps)+1):"),
    ("atomic_torch(ckvalue(args.steps),folder/'last.pt')","if args.stop_at and args.stop_at<args.steps:\n        save(folder/'pilot.json',dict(passed=True,step=args.stop_at,target_steps=args.steps,mechanical_checks_only=True))\n        return\n    atomic_torch(ckvalue(args.steps),folder/'last.pt')"),
    ]
    for old,new in replacements:t=replace(t,old,new)
    trainer=snap/'train_module_v49.py'
    if trainer.exists():assert trainer.read_text()==t
    else:trainer.write_text(t)
    # Transform only observations when replaying the actual trained last checkpoints.
    t=(CODE/'infer_online_diagnostic_v48.py').read_text()
    replacements=[
    ("from train_online_parameter_v48 import RUN","from train_module_v49 import RUN\nfrom module_training_model_v49 import ARMS,transform_input"),
    ("OUT=RUN/'diagnostic_v46';INPUT=SOURCE/'diagnostic_v46'","OUT=RUN/'diagnostic_v46';INPUT=SOURCE/'diagnostic_v46'"),
    ("choices=['initial','frozen_last','joint_last','protected_last','protected_best']","choices=ARMS"),
    ("for mode in ['joint','frozen','protected']:assert json.loads((RUN/mode/'done.json').read_text())['complete']","assert json.loads((RUN/a.mode/'done.json').read_text())['complete']"),
    ("if a.mode!='initial':\n        arm=a.mode.split('_')[0];name='last.pt' if a.mode.endswith('_last') else 'best.pt'\n        path=RUN/arm/name;ck=torch.load(path,weights_only=False,map_location=device)\n        model.load_state_dict(ck['model']);visual.load_tail(ck['visual_tail'])\n    else:path=initial_path;ck=initial","path=RUN/a.mode/'last.pt';ck=torch.load(path,weights_only=False,map_location=device)\n    model.load_state_dict(ck['model']);visual.load_tail(ck['visual_tail'])"),
    ("b['rgb_native']=bank[data['feature_ids'][ix]]","b['rgb_native']=bank[data['feature_ids'][ix]];b=transform_input(b,a.mode)"),
    ]
    for old,new in replacements:t=replace(t,old,new)
    inference=snap/'infer_module_v49.py'
    if inference.exists():assert inference.read_text()==t
    else:inference.write_text(t)
    hashes={}
    for src in CODE.glob('*.py'):
        dest=snap/src.name
        if dest.exists():assert sha(dest)==sha(src),str(src)
        else:dest.write_bytes(src.read_bytes())
        hashes[src.name]=sha(src)
    hashes.update(train_module_v49=sha(trainer),infer_module_v49=sha(inference));(RUN/'code_hashes.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(dict(prepared=True,path=str(RUN),frozen=True)))
if __name__=='__main__':main()
