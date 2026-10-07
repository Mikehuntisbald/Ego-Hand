import subprocess,sys,time
from pathlib import Path
run=Path('/mnt/why/HOT3D/experiments/dit_lowconfidence_v1');project=Path('/mnt/why/hot3d_hand_residual')
while not (run/'temporal_context_done.json').exists():time.sleep(20)
processes=[]
for kind,device in [('dit','cuda:2'),('regression','cuda:3')]:
    log=(run/f'temporal_train_{kind}.log').open('w')
    p=subprocess.Popen([sys.executable,str(project/'train_residual_proof.py'),'--kind',kind,'--device',device,'--temporal','--gate-steps','0'],stdout=log,stderr=subprocess.STDOUT)
    processes.append((kind,device,p,log))
for kind,device,p,log in processes:
    assert p.wait()==0;log.close()
gates=[]
for kind,device in [('dit','cuda:2'),('regression','cuda:3')]:
    log=(run/f'temporal_selective_{kind}.log').open('w')
    p=subprocess.Popen([sys.executable,str(project/'train_selective_gate.py'),'--kind',kind,'--device',device,'--temporal'],stdout=log,stderr=subprocess.STDOUT)
    gates.append((p,log))
for p,log in gates:assert p.wait()==0;log.close()
(run/'temporal_models_done.txt').write_text('All models selected on P0003 only. Ready for locked test.')
