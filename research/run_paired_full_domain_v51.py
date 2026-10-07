"""Finite second iteration: independent interaction groups and real finger GT."""
import fcntl,hashlib,json,os,shutil,subprocess,time,traceback
from pathlib import Path
from run_full_domain_v51 import gpu_free

CODE=Path('/mnt/why/hot3d_hand_residual');RUN=Path('/mnt/why/HOT3D/experiments/full_model_gloves_multihand_v51_20261007')
PYTHON='/mnt/why/HOT3D/experiments/yolo26_wilor_3d_20261003/venv/bin/python'
def status(stage,**data):
    p=RUN/'paired_controller_status.json';temp=p.with_suffix('.tmp');temp.write_text(json.dumps(dict(stage=stage,pid=os.getpid(),time=time.time(),**data),indent=2));os.replace(temp,p)

def main():
    lock=(RUN/'controller.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('An existing owned controller is active; do not duplicate it')
    snapshot=RUN/'source_snapshot_paired'
    if not snapshot.exists():
        snapshot.mkdir();names=['prepare_domain_v51.py','prepare_paired_domain_v51.py','cache_domain_masks_v51_pair.py','build_surgical_pose_cache_v51.py','train_full_surgical_v51_r2.py','train_full_instance_v51_r1.py','train_paired_detector_v51.py','train_domain_detector_v51.py','instance_parameter_model_v51.py','instance_parameter_model_v51_r1.py','instance_parameter_model_v51_r2.py','instance_masks_v51.py','complete_instance_v51.py','complete_instance_v51_r1.py','annotate_instances_v51.py','annotate_instances_v51_r1.py','cache_instance_conditions_v51.py','run_paired_full_domain_v51.py']
        for name in names:shutil.copy2(CODE/name,snapshot/name)
        (RUN/'paired_source_manifest.json').write_text(json.dumps({n:hashlib.file_digest((snapshot/n).open('rb'),'sha256').hexdigest() for n in names},indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',PYTHONPATH=str(snapshot)+':'+str(CODE));paired=RUN/'paired_protocol'
    try:
        while not Path('/mnt/why/HOT3D/domain_data_v51/surgical_hands/selected_provenance.json').exists():
            if not Path('/proc/'+str(json.loads((RUN/'surgical_images_pid.json').read_text())['pid'])).exists():raise RuntimeError('Surgical subset acquisition exited before completion; inspect retained surgical_images.log')
            status('waiting_real_finger_images');time.sleep(20)
        tasks=[('paired_dataset','prepare_paired_domain_v51.py',[],paired/'dataset_protocol.json',False),('paired_mask_cache','cache_domain_masks_v51_pair.py',['--per-group','4'],paired/'domain_masks_done.json',True),('glove_keypoint_cache','build_surgical_pose_cache_v51.py',[],RUN/'surgical_pose/cache_done.json',True),('full_RGB_3D_glove_pilot','train_full_surgical_v51_r2.py',['--steps','160','--resume'],paired/'core_r1/dit_joint/done.json',True),('balanced_full_detector','train_paired_detector_v51.py',[],paired/'detector/done.json',True)]
        for stage,file,args,done,gpu in tasks:
            if done.exists():continue
            if gpu:
                while not gpu_free():status('waiting_GPU3',pending=stage);time.sleep(20)
            with (RUN/f'{stage}.log').open('ab') as log:
                child=subprocess.Popen([PYTHON,'-u',str(snapshot/file),*args],cwd=CODE,env=env,stdout=log,stderr=subprocess.STDOUT)
                status(stage,child_pid=child.pid);exitcode=child.wait()
            if exitcode:raise RuntimeError(f'{stage} exited {exitcode}; source and logs retained')
            assert done.exists()
        status('paired_training_complete',full_pipeline_test_pending=True,default_changed=False)
    except Exception as error:
        (RUN/'paired_controller_error.txt').write_text(traceback.format_exc());status('error',error=repr(error));raise

if __name__=='__main__':main()
