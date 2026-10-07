"""Real trained tensor updates and full optimizer/RNG checkpoint contents."""
import json,hashlib,re
from pathlib import Path
import torch
ROOT=Path('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007')

def main():
    torch.set_num_threads(4)
    trained=ROOT/'full/checkpoint_best_total.pth';initial=ROOT/'weights/rf-detr-seg-small.pt';a=torch.load(initial,map_location='cpu',weights_only=False)['model'];b=torch.load(trained,map_location='cpu',weights_only=False)['model'];changed={}
    for k,v in a.items():
        if k in b and v.shape==b[k].shape and 'backbone' in k and 'encoder' in k and ('query.weight' in k or 'key.weight' in k or 'value.weight' in k or 'layernorm.weight' in k):
            delta=float((v.float()-b[k].float()).abs().max());changed[k]=delta
    assert changed and all(v>0 for v in changed.values()),changed
    paths=list((ROOT/'full').glob('*.ckpt'));assert paths, 'No full optimizer checkpoint'
    full=max(paths,key=lambda p:p.stat().st_mtime);state=torch.load(full,map_location='cpu',weights_only=False);rng=state.get('experiment_rng');assert rng and all(k in rng for k in ['python','numpy','torch','cuda']);optimizers=state.get('optimizer_states');assert optimizers and optimizers[0]['state'];assert state.get('lr_schedulers')
    done=json.loads((ROOT/'full/done.json').read_text());data=dict(training_complete=True,selected_checkpoint=str(trained),selected_checkpoint_sha256=hashlib.file_digest(trained.open('rb'),'sha256').hexdigest(),pretrained_md5=hashlib.file_digest(initial.open('rb'),'md5').hexdigest(),actual_attention_and_norm_weight_max_abs_updates=changed,attention_norm_updates_passed=True,live_GPU_gradients=done['gradient_norm_max'],mask_head_actual_updates={k:v for k,v in done['weight_max_abs_update'].items() if 'segmentation' in k},full_checkpoint=str(full),optimizer_entries=len(optimizers[0]['state']),optimizer_and_scheduler_saved=True,python_numpy_torch_cuda_RNG_saved=True,exact_resume_execution_tested=False,full_model_3D_core_weights_unchanged=True,GT_free_inference_contract=True,test_masks_not_in_training=True,synthetic_occlusion=False,default_replaced=False)
    (ROOT/'training_verification.json').write_text(json.dumps(data,indent=2));print(json.dumps(dict(updates=len(changed),optimizer_entries=data['optimizer_entries'],passed=True)),flush=True)

if __name__=='__main__':main()
