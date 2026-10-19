import json,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/why/native_candidate_sequence_v10_20261019')
def main():
    assert json.loads((R/'full_status.json').read_text())['complete'];assert not (R/'finish_status.json').exists()
    for name in ['prepare_evaluators.py','evaluate_module.py','evaluate_full.py','verify_final.py']:
        (R/'finish_status.json').write_text(json.dumps(dict(active=True,stage=name,pid=__import__('os').getpid(),unix=time.time()),indent=2))
        with (R/'logs'/name.replace('.py','.log')).open('w') as log:subprocess.run([sys.executable,str(R/'eval_src'/name)],stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(stage=name,complete=True)),flush=True)
    (R/'finish_status.json').write_text(json.dumps(dict(active=False,complete=True,production_admitted=False),indent=2))
if __name__=='__main__':main()
