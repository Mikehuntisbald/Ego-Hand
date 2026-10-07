"""After all clips arrive: perform final audit and refresh the offline report."""
import json,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/why/HOT3D');project=Path('/mnt/why/hot3d_hand_residual')
def status(stage,**extra):
    data=dict(stage=stage,**extra);part=root/'completion_status.json.partial'
    part.write_text(json.dumps(data,indent=2));part.replace(root/'completion_status.json')
status('waiting_for_download')
while True:
    d=json.loads((root/'download_status.json').read_text())
    if d['stage']=='failed':status('failed',reason='Downloader ended with errors',errors=d['errors']);raise SystemExit(1)
    if d['stage']=='complete':break
    time.sleep(20)
status('waiting_for_export')
while True:
    e=json.loads((root/'export_status.json').read_text())
    if e['stage']=='complete' and e['completed_clips']==460:break
    time.sleep(20)
try:
    status('auditing')
    subprocess.run([sys.executable,str(project/'audit_subset.py'),'--final'],check=True)
    subprocess.run([sys.executable,str(project/'build_report.py')],check=True)
    # Separate du invocations: one invocation with parent and child operands
    # otherwise counts the parent after excluding the already visited children.
    space=''.join(subprocess.check_output(['du','-sh',str(p)],text=True) for p in [root/'rgb_clips',root/'export',root])
    status('complete',frames=69000,train_frames=57000,val_frames=12000,storage=space,report=str(root/'report.html'))
    print((root/'completion_status.json').read_text(),flush=True)
except Exception as ex:
    status('failed',reason=repr(ex));raise
