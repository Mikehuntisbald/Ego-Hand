"""Finite queue/controller: GPU3 only, no cron or recurring automation."""
import argparse,fcntl,json,os,subprocess,time,traceback
from pathlib import Path
from prepare_fair_v54 import ROOT,E,RF,save,sha

BASE=str(E/'yolo26_wilor_3d_20261003/venv/bin/python')
RFPY=str(E/'rfdetr_hand_instance_v52_20261007/venv/bin/python')
UUID='GPU-2f45f698-b39e-0f82-204d-f3a7d925baa9'
def occupied():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader']).decode()
    return [line.strip() for line in out.splitlines() if UUID in line]
def main():
    ROOT.mkdir(exist_ok=True);lock=(ROOT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (ROOT/'done.json').exists():print((ROOT/'done.json').read_text());return
    started=time.time();env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(ROOT/'code'))
    def status(stage,**kw):save(ROOT/'controller_status.json',dict(stage=stage,pid=os.getpid(),timestamp=time.time(),elapsed_s=time.time()-started,finite=True,automation_restarted=False,default_changed=False,**kw))
    def hashes():
        expected=json.loads((ROOT/'code_hashes.json').read_text())
        changed=[name for name,h in expected.items() if sha(ROOT/'code'/name)!=h];assert not changed,changed
    def run(name,script,args=(),rf=False,gpu=True):
        hashes()
        if (ROOT/'STOP').exists():raise RuntimeError('STOP requested; this experiment only')
        if gpu:
            while True:
                apps=occupied()
                if not apps:break
                status('waiting_GPU3',next_stage=name,other_GPU3_processes=apps)
                if time.time()-started>24*3600:raise TimeoutError('Finite controller queue/runtime reached 24 hours')
                time.sleep(30)
                if (ROOT/'STOP').exists():raise RuntimeError('STOP requested while queued')
        status(name)
        log_path=ROOT/'logs'/f'{name}.log';log_path.parent.mkdir(exist_ok=True)
        with log_path.open('a') as log:
            command=[RFPY if rf else BASE,str(ROOT/'code'/script),*map(str,args)]
            proc=subprocess.Popen(command,env=env,cwd=ROOT/'code',stdout=log,stderr=subprocess.STDOUT);status(name,child_pid=proc.pid,log=str(log_path),command=command)
            while proc.poll() is None:
                time.sleep(10)
                if (ROOT/'STOP').exists():proc.terminate();proc.wait(timeout=30);raise RuntimeError('STOP requested; own child stopped')
            if proc.returncode:raise RuntimeError(f'{name} exited {proc.returncode}; logs and snapshot retained')
    def predict(name,manifest,folder):
        if not (folder/'freeze.json').exists():run(name,'predict_instances_v53.py',['--input',manifest,'--output',folder,'--checkpoint',RF,'--threshold',.2,'--policy','box_nms_0.7'],rf=True)
    try:
        assert json.loads((ROOT/'prepared.json').read_text())['complete'];hashes()
        assert json.loads((ROOT/'matched_training_verification.json').read_text())['passed']
        clips=json.loads((ROOT/'replay_clips.json').read_text());frames=[]
        for c in clips:frames+=json.loads(Path(c['input']).read_text())['frames']
        frames+=json.loads((ROOT/'replay/nail.json').read_text())['frames'];save(ROOT/'native_replay_rgb.json',dict(frames=frames))
        predict('native_replay_RF',ROOT/'native_replay_rgb.json',ROOT/'replay_RF');predict('glove_replay_RF',ROOT/'glove_replay_rgb.json',ROOT/'glove_RF')
        tags=[c['id'] for c in clips]+['glove','nail']
        for front in ['yolo','rf','yolo_control']:
            for tag in tags:
                if not (ROOT/'sources'/front/tag/'freeze.json').exists():run(f'source_{front}_{tag}','infer_fair_v54.py',['source','--front',front,'--tag',tag])
        from infer_fair_v54 import MODES
        # Every model output must finish and freeze before the scorer opens GT.
        for mode in MODES:
            for tag in tags:
                if not (ROOT/'inference'/mode/tag/'freeze.json').exists():run(f'infer_{mode}_{tag}','infer_fair_v54.py',['predict','--mode',mode,'--tag',tag])
        if not (ROOT/'summary.json').exists():run('score_and_render','score_fair_v54.py',gpu=False)
        save(ROOT/'done.json',dict(complete=True,report=str(ROOT/'review/report.html'),default_changed=False,automation_restarted=False,seconds=time.time()-started));status('complete')
    except Exception as exc:
        status('needs_attention',error=repr(exc),traceback=traceback.format_exc(),frozen_sources_preserved=True);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--start',action='store_true');p.add_argument('--status',action='store_true');a=p.parse_args()
    if a.status:print((ROOT/'controller_status.json').read_text() if (ROOT/'controller_status.json').exists() else '{}')
    elif a.start:
        ROOT.mkdir(exist_ok=True)
        path=ROOT/'controller_status.json'
        if path.exists():
            previous=json.loads(path.read_text())
            if previous.get('stage') not in ['complete','needs_attention']:
                try:os.kill(previous['pid'],0)
                except ProcessLookupError:pass
                else:print(json.dumps(dict(already_active=True,pid=previous['pid'],stage=previous['stage'])));raise SystemExit(0)
        with (ROOT/'launcher.log').open('a') as log:
            proc=subprocess.Popen([BASE,__file__],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=Path(__file__).parent)
        print(json.dumps(dict(launcher_pid=proc.pid,finite=True,automation_restarted=False)),flush=True)
    else:main()
