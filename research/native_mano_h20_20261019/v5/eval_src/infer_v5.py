"""Right-only native heads. No hand annotations/targets may be opened."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_mano_supervision_v5_20261019');V=Path('/mnt/why/native_mano_multiseq_v3_20261019')
sys.path.insert(0,str(V/'src'));sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src')
import infer_validation as engine
from native_mano import NativeMANO
from native_dit import NativeMANODiT
ARMS=['native_regression','proxy_dit','native_dit','native_protected']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def predict_right(model,codec,b,seed=190019,zero_rgb=False):
    samples=model.sample(b,draws=4,steps=10,seed=seed,zero_rgb=zero_rgb)
    delta=samples.mean(0)*b['right'].float()[:,None,None]
    return codec.apply_residual(b['state'],b['betas'],b['right'],delta),delta,samples

def forbid(event,args):
    engine.forbid_hand_labels(event,args)
    if event=='open' and isinstance(args[0],(str,bytes)):
        assert 'native_targets.pt' not in str(args[0]),'Native labels forbidden at inference'

def main():
    torch.set_num_threads(4);sys.addaudithook(forbid);engine.predict=predict_right
    assert json.loads((R/'controller_status.json').read_text())['complete']
    protocol=json.loads((V/'protocol.json').read_text());models={a:R/'runs'/a/'best_development.pt' for a in ARMS}
    lock={a:dict(path=str(p),sha256=sha(p)) for a,p in models.items()}
    (R/'inference_lock.json').write_text(json.dumps(dict(models=lock,locked=time.time(),right_only=True,existing_reference_camera=True,
        historical_exposure='All six clips were evaluated in v3/v4 and the posthoc geometry audit before v5 design; diagnostic development evidence, not a fresh blind test',
        candidate_visualization='native_protected_guard predeclared',GT_input=False),indent=2))
    codec=NativeMANO();seals={};start=time.time()
    for clip in protocol['clips']:
        view=Path(clip['view']);data=engine.Observations(view);out=R/'predictions'/f"clip-{clip['clip']:06}";out.mkdir(parents=True,exist_ok=False)
        for arm,p in models.items():
            assert sha(p)==lock[arm]['sha256'];ck=torch.load(p,weights_only=False,map_location='cuda');model=NativeMANODiT(**ck['config']).cuda().eval();model.load_state_dict(ck['model'])
            for guard in [False,True]:
                value,parity=engine.run_model(model,codec,data,guard=guard)
                left=~data.right.bool();assert np.array_equal(value['normalized_residual'][left.cpu().numpy()],np.zeros_like(value['normalized_residual'][left.cpu().numpy()]))
                np.savez_compressed(out/(arm+('_guard' if guard else '')+'.npz'),**value)
                assert parity<1e-6
            del model
        payload=dict(complete=True,clip=clip['clip'],sequence=clip['sequence'],files={p.name:sha(p) for p in out.glob('*.npz')},hand_GT_read=False,native_export_parity_max_m=parity)
        seal=out/'seal.json';seal.write_text(json.dumps(payload,indent=2));seals[str(seal.relative_to(R))]=sha(seal)
        del data;torch.cuda.empty_cache();print(json.dumps(dict(clip=clip['clip'],complete=True,seconds=time.time()-start)),flush=True)
    (R/'predictions_frozen.json').write_text(json.dumps(dict(complete=True,freeze_unix=time.time(),models=lock,seals=seals,hand_labels_read=False,
        inference_source_sha256=sha(Path(__file__)),scope='new outputs frozen before this run labels; prior evaluation exposure explicitly acknowledged'),indent=2))

if __name__=='__main__':main()
