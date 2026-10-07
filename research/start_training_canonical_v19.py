"""Run independent control while risk refits; train canonical only afterward."""
import json, subprocess, time
from pathlib import Path
from hand3d_v8_common import V7, save

CODE = Path(__file__).resolve().parent
PYTHON = '/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'
RUN = V7.parent / 'canonical_native_v19'
RUN.mkdir(exist_ok=True)

def start(name, stage, variant, device):
    log = (RUN / f'{name}.log').open('a')
    proc = subprocess.Popen([PYTHON, str(CODE / 'train_canonical_rgb_v19.py'), '--stage', stage,
        '--variant', variant, '--device', device], stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return proc

def main():
    if (RUN / 'training_job.json').exists():
        old = json.loads((RUN / 'training_job.json').read_text())
        pid = old['pid']
        cmd = Path(f'/proc/{pid}/cmdline')
        assert not (cmd.exists() and b'start_training_canonical_v19.py' in cmd.read_bytes()), 'Already running'
    save(RUN / 'training_job.json', dict(pid=__import__('os').getpid(), stage='risk and matched control'))
    control = start('control', 'model', 'control', 'cuda:1')
    risk = start('risk', 'risk', 'canonical', 'cuda:0')
    save(RUN / 'training_status.json', dict(control_pid=control.pid, risk_pid=risk.pid, stage='risk and control'))
    assert risk.wait() == 0, 'Canonical risk failed; inspect risk.log'
    canonical = start('canonical', 'model', 'canonical', 'cuda:2')
    save(RUN / 'training_status.json', dict(control_pid=control.pid, canonical_pid=canonical.pid, stage='matched models'))
    assert canonical.wait() == 0, 'Canonical model failed; inspect canonical.log'
    assert control.wait() == 0, 'Control model failed; inspect control.log'
    save(RUN / 'training_status.json', dict(complete=True, stage='Both model runs terminal; development evaluation required'))

if __name__ == '__main__':
    main()
