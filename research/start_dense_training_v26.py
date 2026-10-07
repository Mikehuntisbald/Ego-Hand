import json,subprocess,os
from pathlib import Path
from hand3d_v8_common import V7,save

def main():
    code=Path(__file__).resolve().parent;run=V7.parent/'dense_training_v26';run.mkdir(exist_ok=True)
    assert json.loads((V7.parent/'dense_centers_v26/ready.json').read_text())['complete']
    assert not (run/'training_job.json').exists(),'Existing job: inspect before restarting'
    save(run/'training_job.json',dict(pid=os.getpid(),stage='Matched6000updates and originaldevelopment; first refit expandedsubjectOOF risks'))
    def start(name,stage,variant,device):
        log=(run/f'{name}.log').open('a')
        return subprocess.Popen(['/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python',str(code/'train_dense_centers_v26.py'),'--variant',variant,'--stage',stage,'--device',device],cwd=str(code),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    control=start('control','model','control','cuda:1');risk=start('risk','risk','expanded','cuda:0')
    save(run/'training_status.json',dict(control_pid=control.pid,risk_pid=risk.pid,stage='Risk and control'))
    assert risk.wait()==0,'Expandedrisk failed; inspect risk.log'
    candidate=start('expanded','model','expanded','cuda:2')
    save(run/'training_status.json',dict(control_pid=control.pid,expanded_pid=candidate.pid,stage='Matched models6000updates'))
    assert candidate.wait()==0,'Expanded model failed'
    assert control.wait()==0,'Control model failed'
    save(run/'training_status.json',dict(complete=True,stage='Bothmodelruns terminal; development evaluation required'))

if __name__=='__main__':main()
