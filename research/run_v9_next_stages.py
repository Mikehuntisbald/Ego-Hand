import json,subprocess,sys
from pathlib import Path
from hand3d_v8_common import save
root=Path(__file__).resolve().parent;experiments=Path('/mnt/why/HOT3D/experiments');jobs=[]
for kind in ['dit','regression']:
    folder=experiments/'offline_hand3d_v9_rollout';folder.mkdir(exist_ok=True);log=open(folder/f'{kind}.log','w');proc=subprocess.Popen([sys.executable,str(root/'train_trajectory_rollout_v9.py'),'--kind',kind,'--device','cuda:3'],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);log.close();jobs.append(dict(stage='trajectory_rollout',kind=kind,pid=proc.pid,gpu=3))
for fold in range(3):
    folder=experiments/'proposal_critic_v9';folder.mkdir(exist_ok=True);log=open(folder/f'fold{fold}.log','w');proc=subprocess.Popen([sys.executable,str(root/'train_proposal_oof_v9.py'),'--fold',str(fold),'--device',f'cuda:{fold}'],cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True);log.close();jobs.append(dict(stage='proposal_oof',fold=fold,pid=proc.pid,gpu=fold))
save(experiments/'offline_hand3d_v9/next_stage_processes.json',jobs);print(json.dumps(jobs),flush=True)
