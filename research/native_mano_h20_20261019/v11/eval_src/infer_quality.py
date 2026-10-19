"""Freeze six learned selectors on existing v10 proposals; GT access forbidden."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_candidate_risk_v11_20261019');V10=Path('/mnt/why/native_candidate_sequence_v10_20261019');V3=Path('/mnt/why/native_mano_multiseq_v3_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019')
sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R/'error_only_r1/src'));sys.path.insert(0,str(V3/'src'))
from risk_model import NativeQuality as Initial
from risk_model_r1 import NativeQuality as ErrorOnly
from infer_validation import Observations
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forbid(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=str(args[0]);assert not any(x in name for x in ['.hands.json','__hand_shapes.json__','native_targets.pt','quality_targets.pt','/targets.pt','scoring_arrays.npz','capacity_diagnostic.json']), 'Hand-label access forbidden during quality deployment'
def main():
    torch.set_num_threads(4);torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);sys.addaudithook(forbid);assert json.loads((R/'error_only_r1/controller_status.json').read_text())['complete'];models={}
    for prefix,base,kind in [('initial',R,Initial),('error',R/'error_only_r1',ErrorOnly)]:
        for arm in ['pose_base','RGB_base','RGB_preserve']:models[prefix+'_'+arm]=(base/'runs'/arm/'best_development.pt',kind)
    lock={a:sha(p) for a,(p,_) in models.items()};(R/'inference_lock.json').write_text(json.dumps(dict(complete=True,models=lock,GT_input=False,source_sha256=sha(Path(__file__))),indent=2));clips=json.loads((V3/'protocol.json').read_text())['clips']+[dict(clip=2102,view=str(V2/'coherent_frontend'))];seals={};keys=['state_camera51','joints_camera21','normalized_residual','protection_alpha','global_orient','hand_pose','betas','transl','reflection','wrist_camera','root_rotation_physical'];metadata=['track_id','timestamp_ns','frame_index','right']
    for clip in clips:
        ident=int(clip['clip']);assert 0<=ident<1000000;view=Path(clip['view']);assert view.resolve().is_relative_to(Path('/mnt/why'));data=Observations(view);bank_path=V10/f'candidates/clip-{ident:06}/candidates.npz';seal=json.loads((bank_path.parent/'seal.json').read_text());assert sha(bank_path)==seal['files']['candidates.npz'];bank=np.load(bank_path);N=len(bank['right']);assert N==data.N;folder=R/f'predictions/clip-{ident:06}';folder.mkdir(parents=True,exist_ok=False);candidate={k:torch.tensor(bank[k],device='cuda') for k in ['joints_camera21','normalized_residual','crop_uv']};reports={}
        for arm,(path,kind) in models.items():
            assert sha(path)==lock[arm];ck=torch.load(path,map_location='cpu',weights_only=False);model=kind(width=96,use_rgb=ck['use_rgb']).cuda().eval();model.load_state_dict(ck['model']);parts={}
            with torch.inference_mode():
                for start in range(0,N,8):
                    end=min(N,start+8);b=data.batch(start,end);c={k:v[start:end] for k,v in candidate.items()}
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b,c)
                    assert torch.isfinite(out['score']).all() and (out['score'][:,0]==0).all()
                    for k in ['score','delta_error_mm','harm_probability','reference_good_logit']:parts.setdefault(k,[]).append(out[k].float().cpu().numpy())
            score={k:np.concatenate(v) for k,v in parts.items()};choice=score['score'].argmin(1);output={k:bank[k][np.arange(N),choice] for k in keys};output.update({k:bank[k] for k in metadata});output.update(candidate_index=choice,quality_score=score['score'],predicted_error_difference_mm=score['delta_error_mm'],predicted_harm_probability=score['harm_probability']);assert np.array_equal(output['wrist_camera'],bank['wrist_camera'][:,0]) and np.array_equal(output['betas'],bank['betas'][:,0]);np.savez_compressed(folder/f'{arm}.npz',**output);reports[arm]=dict(candidate_counts=np.bincount(choice,minlength=6).tolist(),checkpoint_sha256=lock[arm],GT_input=False);del model
        payload=dict(complete=True,GT_used=False,files={p.name:sha(p) for p in folder.glob('*.npz')},arms=reports,candidate_sha256=sha(bank_path));(folder/'seal.json').write_text(json.dumps(payload,indent=2));seals[str((folder/'seal.json').relative_to(R))]=sha(folder/'seal.json');print(json.dumps(dict(clip=ident,candidate_counts={a:v['candidate_counts'] for a,v in reports.items()})),flush=True);del data;torch.cuda.empty_cache()
    (R/'predictions_frozen.json').write_text(json.dumps(dict(complete=True,freeze_unix=time.time(),seals=seals,models=lock,GT_used=False,production_admitted=False),indent=2))
if __name__=='__main__':main()
