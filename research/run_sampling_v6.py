import subprocess,sys,json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
R=Path('/mnt/why/HOT3D/experiments/temporal_sampling_v6');C=Path('/mnt/why/hot3d_hand_residual/temporal_sampling_v6.py')
def worker(device,jobs):
    results={}
    for kind,pattern in jobs:
        name=f'{kind}_{pattern}'
        with open(R/f'{name}.log','w') as f:
            results[name]=subprocess.call([sys.executable,str(C),'train','--kind',kind,'--pattern',pattern,'--device',f'cuda:{device}'],stdout=f,stderr=subprocess.STDOUT,cwd=C.parent)
        if results[name]:break
    return results
with ThreadPoolExecutor(max_workers=4) as pool:
    tasks=[pool.submit(worker,0,[('regression','uniform'),('regression','wide')]),pool.submit(worker,1,[('dit','uniform'),('dit','wide')]),pool.submit(worker,2,[('regression','multiscale')]),pool.submit(worker,3,[('dit','multiscale')])]
    status={k:v for task in tasks for k,v in task.result().items()}
(R/'status.json').write_text(json.dumps(status))
