"""Additional trained arms are sealed before the six-recording hand labels open."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_mano_rollout_v4_20261019');VALIDATION=Path('/mnt/why/native_mano_multiseq_v3_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019')
sys.path.insert(0,str(VALIDATION/'src'))
from infer_validation import Observations,run_model,forbid_hand_labels
from native_mano import NativeMANO
from native_dit import NativeMANODiT

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4);sys.addaudithook(forbid_hand_labels);assert not (VALIDATION/'evaluation').exists()
    protocol=json.loads((VALIDATION/'protocol.json').read_text());assert json.loads((R/'receipts/done.json').read_text())['complete']
    models={f'rollout_{arm}':R/f'runs/{arm}/best_development.pt' for arm in ['dit','protected']}
    model_lock={name:dict(path=str(p),sha256=sha(p)) for name,p in models.items()}
    (R/'inference_lock.json').write_text(json.dumps(dict(complete=True,locked_unix=time.time(),models=model_lock,reference_validation_protocol_sha256=sha(VALIDATION/'protocol.json'),hand_labels_unopened=True,
        additional_arm='rollout_dit_displacement uses exactly the existing 5/10/30mm parameter guard',same_deployment_sampler=True),indent=2))
    codec=NativeMANO();seals={};start=time.time()
    for clip in protocol['clips']:
        view=Path(clip['view']);data=Observations(view);out=view/'predictions_rollout';out.mkdir(exist_ok=False)
        for name,p in models.items():
            assert sha(p)==model_lock[name]['sha256'];ck=torch.load(p,weights_only=False,map_location='cuda');model=NativeMANODiT(**ck['config']).cuda().eval();model.load_state_dict(ck['model'])
            value,parity=run_model(model,codec,data);np.savez_compressed(out/f'{name}.npz',**value)
            if name=='rollout_dit':value,parity=run_model(model,codec,data,guard=True);np.savez_compressed(out/'rollout_dit_displacement.npz',**value)
            del model
        payload=dict(complete=True,source_clip=clip['clip'],sequence=clip['sequence'],hand_GT_read=False,reference_camera_used=True,files={p.name:sha(p) for p in out.glob('*.npz')})
        path=out/'seal.json';path.write_text(json.dumps(payload,indent=2));seals[str(path.relative_to(VALIDATION))]=sha(path)
        print(json.dumps(dict(clip=clip['clip'],complete=True,seconds=time.time()-start)),flush=True);del data;torch.cuda.empty_cache()
    seal=dict(complete=True,freeze_unix=time.time(),hand_GT_read_before_freeze=False,seals=seals,models=model_lock,original_prediction_manifest_sha256=sha(VALIDATION/'receipts/predictions_frozen.json'),
        source_sha256={p.name:sha(p) for p in (R/'src').glob('*.py')})
    (VALIDATION/'receipts/rollout_predictions_frozen.json').write_text(json.dumps(seal,indent=2))

if __name__=='__main__':main()
