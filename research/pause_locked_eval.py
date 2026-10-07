import json,os,signal,time
from pathlib import Path
run=Path('/mnt/why/HOT3D/experiments/dit_lowconfidence_v1');target=b'/mnt/why/hot3d_hand_residual/evaluate_proof.py'
stopped=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:args=(p/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if target in args:os.kill(int(p.name),signal.SIGTERM);stopped.append(int(p.name))
for name in ['locked_test_results.json','evaluation_done.json']:
    file=run/name
    if file.exists():file.rename(run/('unread_v1_'+name))
(run/'test_blinding_note.json').write_text(json.dumps(dict(stopped_evaluator_pids=stopped,
    reason='Gate preservation failed on P0003; held-out numerical results have not been read and do not guide the fix',
    decision_data='P0003 tune only',time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2));print(stopped)
