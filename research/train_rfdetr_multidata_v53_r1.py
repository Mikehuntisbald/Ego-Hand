"""Official RF-DETR training with experiment-owned audit/RNG callbacks."""
import os, json, time, random, argparse, hashlib, subprocess
from pathlib import Path
import numpy as np
import torch

ROOT=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')

def gpu_free():
    uuid=subprocess.check_output(['nvidia-smi','--id=3','--query-gpu=uuid','--format=csv,noheader']).decode().strip()
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader']).decode()
    return not any(uuid in s for s in apps.splitlines())

def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--epochs',type=int,default=10);p.add_argument('--resume');a=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3', 'Only physical GPU3 is authorized'
    assert gpu_free(), 'GPU3 is occupied; do not start a second training process'
    from rfdetr import RFDETR
    from rfdetr_partial_supervision_v53_r1 import install
    install()
    import rfdetr.training as rt
    from pytorch_lightning import Callback
    from pytorch_lightning import seed_everything
    seed_everything(5207,workers=True);torch.set_num_threads(8)
    out=ROOT/('pilot_r1' if a.pilot else 'full');out.mkdir(exist_ok=True)
    source=Path(__file__);(out/'source.py').write_bytes(source.read_bytes())
    class Audit(Callback):
        def __init__(self):self.start=time.time();self.initial={};self.grad={};self.updates={}
        def on_train_start(self,tr,m):
            for kind in ['encoder','backbone','segmentation','class_embed','bbox_embed']:
                keys=[(n,v) for n,v in m.named_parameters() if kind in n and v.requires_grad and v.numel()>100 and n.endswith("weight")]
                if not keys and kind=='segmentation':keys=[(n,v) for n,v in m.named_parameters() if 'mask' in n and v.requires_grad and v.numel()>100 and n.endswith("weight")]
                if keys:
                    n,v=keys[-1];self.initial[n]=v.detach().float().cpu().clone();print('AUDIT_PARAMETER',kind,n,flush=True)
            assert len(self.initial)>=3, list(self.initial)
        def on_before_optimizer_step(self,tr,m,opt):
            for n,v in m.named_parameters():
                if n in self.initial and v.grad is not None:self.grad[n]=max(self.grad.get(n,0.),float(v.grad.detach().float().norm()))
        def on_train_batch_end(self,tr,m,outputs,batch,batch_idx):
            if tr.global_step%25==0:
                self.write(tr,m)
        def write(self,tr,m):
            current=dict(m.named_parameters())
            self.updates={n:float((current[n].detach().float().cpu()-v).abs().max()) for n,v in self.initial.items()}
            metrics={k:float(v) for k,v in tr.callback_metrics.items() if torch.is_tensor(v) and v.numel()==1}
            data=dict(step=tr.global_step,epoch=tr.current_epoch,elapsed_s=time.time()-self.start,gradient_norm_max=self.grad,weight_max_abs_update=self.updates,metrics=metrics,GT_mask_loss_only=True,no_generated_occlusion=True)
            (out/'progress.json').write_text(json.dumps(data,indent=2))
        def on_save_checkpoint(self,tr,m,ckpt):
            ckpt['experiment_rng']=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())
        def on_load_checkpoint(self,tr,m,ckpt):
            state=ckpt.get('experiment_rng');assert state is not None, 'Missing exact experiment RNG state'
            random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state_all(state['cuda'])
        def on_train_end(self,tr,m):self.write(tr,m)
    audit=Audit();original=rt.build_trainer
    def build(tc,mc,**kwargs):
        tr=original(tc,mc,**kwargs);tr.callbacks.append(audit);return tr
    rt.build_trainer=build
    model=RFDETR.from_checkpoint('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/full/checkpoint_best_total.pth',device='cuda:0',trust_checkpoint=True)
    config=dict(dataset_dir=str(ROOT/('pilot_data' if a.pilot else 'data')),output_dir=str(out),epochs=1 if a.pilot else a.epochs,batch_size=8 if a.pilot else 16,eval_batch_size=4,grad_accum_steps=1,lr=1e-4,lr_encoder=1e-5,lr_scheduler="cosine",lr_scheduler_kwargs={"lr_drop":5,"min_factor":.1},checkpoint_interval=1,resolution=576,multi_scale=False,expanded_scales=False,scale_jitter=False,aug_config={},amp_dtype='bf16',num_workers=4,use_ema=True,run_test=False,tensorboard=False,wandb=False,mlflow=False,clearml=False,progress_bar=None,save_dataset_grids=False,skip_best_epochs=0,eval_max_dets=100,eval_backend='faster_coco_eval',seed=5207)
    if a.resume:config['resume']=a.resume
    (out/'requested_config.json').write_text(json.dumps(config,indent=2))
    model.train(**config)
    data=json.loads((out/'progress.json').read_text());data['training_complete']=True
    data['visual_and_mask_updates_passed']=all(v>0 for v in audit.grad.values()) and all(v>0 for v in audit.updates.values()) and len(audit.grad)>=3
    assert data['visual_and_mask_updates_passed'],data
    (out/'done.json').write_text(json.dumps(data,indent=2));print('DONE',json.dumps(data),flush=True)

if __name__=='__main__':main()
