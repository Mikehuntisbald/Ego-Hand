import json, subprocess
from pathlib import Path
from hand3d_v8_common import V7, save

def main():
    run = V7.parent/'dual_visual_v20'
    assert json.loads((run/'preflight.json').read_text())['passed']
    assert not (run/'training_job.json').exists(), 'Existing launch; inspect rather than restart'
    code = Path(__file__).resolve().parent
    jobs = {}
    for variant, device in [('control','cuda:1'),('canonical','cuda:2')]:
        log = (run/f'{variant}.log').open('a')
        p = subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python',
            str(code/'train_dual_visual_v20.py'),'--variant',variant,'--device',device],
            cwd=str(code),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs[variant] = dict(pid=p.pid,device=device)
    save(run/'training_job.json',dict(jobs=jobs,stage='Matched dual semantic condition, zero gain initial parity passed'))
    print((run/'training_job.json').read_text())

if __name__=='__main__':
    main()
