import subprocess,sys,json,hashlib,time
from pathlib import Path
ROOT=Path('/mnt/why/HOT3D/experiments/natural_visual_ft_v5');ROOT.mkdir(exist_ok=True)
CODE=Path('/mnt/why/hot3d_hand_residual/finetune_visual_v5.py')
protocol=dict(steps=600,effective_batch=16,arms=['regression_frozen','regression_ft','dit_frozen','dit_ft'],tune='last 4/32 visual blocks + spatial stem + temporal head',selection='dev_select only, initial checkpoint included',evaluation='fixed v4 regression gate across all arms; 47 known hardcases are diagnostic only',seed=202610053,code_sha256=hashlib.sha256(CODE.read_bytes()).hexdigest(),natural_inputs=True,created=time.time())
(ROOT/'protocol.json').write_text(json.dumps(protocol,indent=2))
def stage(commands,names):
    jobs=[]
    for command,name in zip(commands,names):
        log=open(ROOT/f'{name}.log','w');p=subprocess.Popen([sys.executable,str(CODE)]+command,stdout=log,stderr=subprocess.STDOUT,cwd=CODE.parent);jobs.append((p,log,name))
    status={}
    for p,log,name in jobs:status[name]=p.wait();log.close()
    (ROOT/('status_'+names[0]+'.json')).write_text(json.dumps(status))
    if any(status.values()):raise RuntimeError(status)
stage([['cache','--device',f'cuda:{i}','--shard',str(i)] for i in range(4)],[f'cache{i}' for i in range(4)])
stage([['train','--device',f'cuda:{i}','--kind',kind,'--mode',mode] for i,(kind,mode) in enumerate([('regression','ft'),('dit','ft'),('regression','frozen'),('dit','frozen')])],['regression_ft','dit_ft','regression_frozen','dit_frozen'])
(ROOT/'all_done.json').write_text(json.dumps(dict(complete=True,time=time.time())))
