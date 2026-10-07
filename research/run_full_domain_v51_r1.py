"""One bounded, locked experiment controller; no recurring automation."""
import fcntl,hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path

CODE=Path('/mnt/why/hot3d_hand_residual')
RUN=Path('/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007')
PYTHON=Path('/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python')

def status(stage,**values):
    data=dict(stage=stage,pid=os.getpid(),time=time.time(),**values);tmp=RUN/'controller_status.tmp';tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,RUN/'controller_status.json');print(json.dumps(data),flush=True)

def gpu_free():
    uid=subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True).splitlines()
    return not any(x.split(',')[0].strip()==uid for x in lines)

def main():
    RUN.mkdir(exist_ok=True);lock=(RUN/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Existing controller owns this run');return
    snapshot=RUN/'source_snapshot_r1'
    if not snapshot.exists():
        snapshot.mkdir();files=['prepare_domain_v51.py','cache_instance_conditions_v51.py','cache_domain_masks_v51.py','train_full_instance_v51_r1.py','train_domain_detector_v51.py','instance_masks_v51.py','instance_parameter_model_v51_r1.py','instance_parameter_model_v51.py','run_full_domain_v51_r1.py']
        hashes={}
        for f in files:shutil.copy2(CODE/f,snapshot/f);hashes[f]=hashlib.file_digest((snapshot/f).open('rb'),'sha256').hexdigest()
        (RUN/'source_manifest.json').write_text(json.dumps(hashes,indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(snapshot)+':'+str(CODE))
    try:
        while not Path('/mnt/why/HOT3D/domain_data_v51/egohands/provenance.json').exists():
            status('waiting_real_egohands_download');time.sleep(20)
        tasks=[('prepare_real_data','prepare_domain_v51.py',[],RUN/'dataset_protocol.json',False),
               ('hot3d_mask_cache','cache_instance_conditions_v51.py',['--train','48','--dev','24'],RUN/'mask_conditions_done.json',True),
               ('real_domain_mask_cache','cache_domain_masks_v51.py',['--per-group','4'],RUN/'domain_masks_done.json',True),
               ('full_model_pilot','train_full_instance_v51_r1.py',['--steps','40','--resume'],RUN/'core_r1/dit_joint/done.json',True),
               ('detector_domain_pilot','train_domain_detector_v51.py',['--epochs','3'],RUN/'detector/done.json',True)]
        for stage,file,args,done,gpu in tasks:
            if done.exists():continue
            if gpu:
                while not gpu_free():status('waiting_GPU3',pending=stage);time.sleep(20)
            status(stage);log=(RUN/f'{stage}.log').open('ab');child=subprocess.Popen([str(PYTHON),'-u',str(snapshot/file),*args],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
            status(stage,child_pid=child.pid);returncode=child.wait();log.close()
            if returncode:raise RuntimeError(f'{stage} failed: returncode={returncode}; retained {stage}.log and source_snapshot')
            assert done.exists(),f'Missing completion evidence: {done}'
        status('pilot_complete',default_changed=False,full_pipeline_validation_pending=True)
    except Exception as e:
        (RUN/'controller_error.txt').write_text(traceback.format_exc());status('error',error=repr(e));raise

if __name__=='__main__':main()
