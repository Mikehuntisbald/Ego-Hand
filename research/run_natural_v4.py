import subprocess,sys
from pathlib import Path
CODE=Path(__file__).resolve().parent;RUN=Path('/mnt/why/HOT3D/experiments/natural_reliability_v4')
with (RUN/'risk.log').open('w') as log:subprocess.run([sys.executable,'-u','natural_reliability.py'],cwd=CODE,stdout=log,stderr=subprocess.STDOUT,check=True)
jobs=[]
for kind,gpu in [('dit',1),('regression',2)]:
    log=(RUN/f'train_{kind}.log').open('w');p=subprocess.Popen([sys.executable,'-u','train_natural_corrector.py','--kind',kind,'--device',f'cuda:{gpu}','--force'],cwd=CODE,stdout=log,stderr=subprocess.STDOUT);jobs.append((p,log))
for p,log in jobs:
    code=p.wait();log.close();assert code==0
subprocess.run([sys.executable,'-u','evaluate_natural_reliability.py'],cwd=CODE,check=True)
