"""Checkpoint evidence; saved RNG is not a claim of tested exact resume."""
import json,hashlib
from pathlib import Path
import torch
A=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')

def main():
    torch.set_num_threads(4)
    initial=Path('/mnt/why/HOT3D/experiments/rfdetr_hand_instance_v52_20261007/full/checkpoint_best_total.pth')
    selected=A/'full/best_admitted_ema.pth'
    src=torch.load(initial,map_location='cpu',weights_only=False)['model'];dst=torch.load(selected,map_location='cpu',weights_only=False)['model']
    updates={k:float((v.float()-dst[k].float()).abs().max()) for k,v in src.items() if k in dst and v.shape==dst[k].shape and 'encoder' in k and any(s in k for s in ['query.weight','key.weight','value.weight','layernorm.weight'])}
    assert updates and all(v>0 for v in updates.values())
    ck=torch.load(A/'full/checkpoint_9.ckpt',map_location='cpu',weights_only=False)
    assert ck['optimizer_states'][0]['state'] and ck['lr_schedulers']
    assert all(k in ck['experiment_rng'] for k in ['python','numpy','torch','cuda'])
    done=json.loads((A/'full/done.json').read_text());assert done['visual_and_mask_updates_passed']
    out=dict(training_complete=True,steps=done['step'],epochs=10,selected_checkpoint=str(selected),checkpoint_sha256=hashlib.file_digest(selected.open('rb'),'sha256').hexdigest(),encoder_attention_and_norm_actual_updates=updates,live_gradients=done['gradient_norm_max'],head_and_projection_updates=done['weight_max_abs_update'],optimizer_entries=len(ck['optimizer_states'][0]['state']),scheduler_and_RNG_saved=True,exact_resume_execution_tested=False,CPPE_prediction_count_gate_missing_from_original_selector=True,narrow_admitted_not_safe_replacement=True,default_replaced=False)
    (A/'training_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(dict(steps=done['step'],verified_encoder_tensors=len(updates),optimizer_entries=out['optimizer_entries'])),flush=True)

if __name__=='__main__':main()
