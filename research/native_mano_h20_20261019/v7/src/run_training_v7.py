"""First controlled 600-update milestone: 155 vs1330 native right windows."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
R=Path('/mnt/why/native_mano_pool_v7_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=R/'training.lock';fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        assert json.loads((R/'receipts/training_preflight.json').read_text())['complete']
        memory=int(subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip());apps=subprocess.check_output(['nvidia-smi','--id=3','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert memory<100 and not apps,'GPU3 occupied'
        sources={p.name:sha(p) for p in (R/'src').glob('*.py')};(R/'receipts/training_source_seal.json').write_text(json.dumps(sources,indent=2))
        protocol=dict(complete=True,phase='first matched600-update milestone; not full goal completion',arms=dict(small=155,full=1330),dev_select_right=25,
            same_model_and_initialization=True,same_updates=600,same_feature_cache=True,same_native_v5_objective=True,visual_backbone_frozen=True,wrist_and_shape_fixed=True,right_only=True,
            subject_sampler='same 3-distinct-subject schedule each update; uniform center within subject; actual example counts reported, not equal-FLOP claim',
            sampler_change='both v7 arms use subject balancing; v5 weights are reference, not the new exact optimization control',
            input_cache_ready_sha256=sha(R/'data/ready.json'),input_verification_sha256=sha(R/'receipts/input_verification.json'),native_targets_sha256=sha(R/'data/native_targets.pt'),
            selection='same fixed P0003 clip25 right windows, native relative20 mm +20*accurate-point-harm fraction, each200',
            evaluation='freeze both outputs before scoring; existing six clips historically exposed, report native GT and preservation/motion together',
            inference_guard='unchanged5/10/30mm; fullBA point2px reprojection remains unchanged',no_GT_inference=True,no_synthetic_occlusion=True,production_admitted=False)
        (R/'training_protocol.json').write_text(json.dumps(protocol,indent=2))
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='3',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH=f'{R}/src:/mnt/why/native_mano_dit_v2_20261019/src:/mnt/why/native_mano_rollout_v4_20261019/src')
        jobs=[('pilot','full',6),('small','small',600),('full','full',600)]
        for name,data_arm,steps in jobs:
            out=R/'runs'/name;out.mkdir(parents=True,exist_ok=True);cmd=[sys.executable,str(R/'src/train_native_v7.py'),'--arm','dit','--data-arm',data_arm,'--steps',str(steps),'--rollout','--output',str(out)]
            if (out/'last.pt').exists():cmd+=['--resume']
            (R/'training_status.json').write_text(json.dumps(dict(pid=os.getpid(),stage=name,data_arm=data_arm,steps=steps,command=cmd,started=time.time()),indent=2))
            with (out/'console.log').open('a') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            done=json.loads((out/'done.json').read_text());assert done['complete'] and done['FK_buffers_unchanged'] and all(v>0 for v in done['gradient_max'].values()) and all(v>0 for v in done['weight_update_max'].values())
            assert sources=={p.name:sha(p) for p in (R/'src').glob('*.py')},'Frozen training source changed'
        summaries={arm:json.loads((R/'runs'/arm/'done.json').read_text()) for arm in ['small','full']};assert summaries['small']['subject_plan_sha256']==summaries['full']['subject_plan_sha256']
        (R/'training_status.json').write_text(json.dumps(dict(complete=True,first_milestone_only=True,stages=[x[0] for x in jobs],same_subject_plan=True,actual_training_examples={a:summaries[a]['training_examples'] for a in summaries},production_admitted=False),indent=2))
    finally:lock.unlink(missing_ok=True)
if __name__=='__main__':main()
