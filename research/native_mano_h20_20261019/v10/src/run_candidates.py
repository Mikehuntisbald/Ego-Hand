import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/why/native_candidate_sequence_v10_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (R/'controller_status.json').exists();sources={p.name:sha(p) for p in (R/'src').glob('*.py')};(R/'receipts/source_seal.json').write_text(json.dumps(sources,indent=2));env=dict(os.environ,CUDA_VISIBLE_DEVICES='3')
    for script,args in [('preflight.py',[]),('build_candidates.py',[]),('select_candidates.py',[])]:
        for n,h in sources.items():assert sha(R/'src'/n)==h
        (R/'controller_status.json').write_text(json.dumps(dict(active=True,pid=os.getpid(),stage=script,unix=time.time()),indent=2))
        with (R/'logs'/script.replace('.py','.log')).open('w') as log:subprocess.run([sys.executable,str(R/'src'/script),*args],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(stage=script,complete=True)),flush=True)
    (R/'controller_status.json').write_text(json.dumps(dict(active=False,complete=True,all_seven_cases_frozen=True,GT_used=False,production_admitted=False),indent=2))
if __name__=='__main__':main()
