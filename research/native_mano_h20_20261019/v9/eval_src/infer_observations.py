"""GT-free image observation proposals on six diagnostic clips and full replay."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_rgb_observation_v9_20261019');V3=Path('/mnt/why/native_mano_multiseq_v3_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019')
sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(V2/'src'));sys.path.insert(0,str(PUBLIC/'src'))
from rgb_head import RGBObservation
from native_mano import NativeMANO
from geometry import load_frames,camera
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forbid(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=str(args[0]);assert not any(v in name for v in ['.hands.json','__hand_shapes.json__','native_targets.pt','/targets.pt','sealed_scoring_arrays.npz','evaluation_labels.pt']),name
def crop_uv(points,rotation,focal,right):
    local=np.einsum('nji,nkj->nki',rotation,points);uv=local[...,:2]/local[...,2:3]*focal[:,None,None]+127.5;uv[~right,:,0]=255.-uv[~right,:,0]
    return ((uv-np.array([31.5,-.5]))/np.array([192.,256.])).astype(np.float32)
def native_uv(uv,rotation,focal,right,frames,rows):
    pixel=uv*np.array([192.,256.])+np.array([31.5,-.5]);pixel[~right,:,0]=255.-pixel[~right,:,0];ray=np.concatenate([(pixel-127.5)/focal[:,None,None],np.ones((*pixel.shape[:2],1))],-1);ray=np.einsum('nij,nkj->nki',rotation,ray)
    return np.stack([camera(frames[row['frame_index']]['cameras']['214-1']['calibration']).eye_to_window(p) for row,p in zip(rows,ray)]).astype(np.float32)
def main():
    torch.set_num_threads(4);sys.addaudithook(forbid);assert json.loads((R/'controller_status.json').read_text())['complete'];models={m:R/'runs'/m/'best_development.pt' for m in ['spatial','pooled']};lock={m:sha(p) for m,p in models.items()}
    (R/'inference_lock.json').write_text(json.dumps(dict(complete=True,models=lock,GT_inputs=False,full_replay='existing clip2102 predicted head/depth unchanged; head observations are new',historical_exposure=True),indent=2))
    clips=json.loads((V3/'protocol.json').read_text())['clips']+[dict(clip=2102,view=str(V2/'coherent_frontend'),sequence='historical-full-replay')];codec=NativeMANO();seals={}
    for clip in clips:
        view=Path(clip['view']);data=np.load(view/'outputs/wilor_clip2102/observations.npz');rows=json.loads((view/'outputs/wilor_clip2102/observations.json').read_text())['frames'];frames=load_frames(view/'data/clip-002102');rights=data['right'].astype(bool);rot=data['crop_rotations'];focal=data['crop_focals'];xyz=codec.from_state(torch.tensor(data['state'],device='cuda'),torch.tensor(data['betas'],device='cuda'),torch.tensor(rights,device='cuda')).detach().cpu().numpy();coarse=crop_uv(xyz,rot,focal,rights)
        old=np.asarray([r['xy'] for r in rows],np.float32);folder=R/'predictions'/f"clip-{clip['clip']:06}";folder.mkdir(parents=True,exist_ok=False);rgb=torch.tensor(data['rgb_spatial'],device='cuda');condition=torch.tensor(coarse,device='cuda');roundtrip=native_uv(coarse,rot,focal,rights,frames,rows);reference=np.stack([camera(frames[r['frame_index']]['cameras']['214-1']['calibration']).eye_to_window(p) for r,p in zip(rows,xyz)]);parity=float(np.abs(roundtrip-reference).max());assert parity<.01,parity
        for mode,p in models.items():
            assert sha(p)==lock[mode];ck=torch.load(p,map_location='cpu',weights_only=False);model=RGBObservation(mode=mode).cuda().eval();model.load_state_dict(ck['model']);parts={}
            with torch.inference_mode():
                for start in range(0,len(rows),32):
                    with torch.autocast('cuda',dtype=torch.bfloat16):out=model(rgb[start:start+32],condition[start:start+32])
                    for k in ['uv','inband_logit','coarse_bad_logit','new_bad_logit']:parts.setdefault(k,[]).append(out[k].float().cpu().numpy())
            result={k:np.concatenate(v) for k,v in parts.items()};sigmoid=lambda x:1/(1+np.exp(-x));weight=.25*sigmoid(result['inband_logit'])*sigmoid(result['coarse_bad_logit'])*(1-sigmoid(result['new_bad_logit']));weight[~rights]=0;uv=native_uv(result['uv'],rot,focal,rights,frames,rows);blended=old+weight[...,None]*(uv-old)
            np.savez_compressed(folder/f'{mode}.npz',uv_head_native=uv,uv_blended_native=blended,uv_baseline_native=old,coarse_fk_crop_norm=coarse,crop_uv=result['uv'],blend_weight=weight,inband_probability=sigmoid(result['inband_logit']),coarse_bad_probability=sigmoid(result['coarse_bad_logit']),new_bad_probability=sigmoid(result['new_bad_logit']),right=rights,timestamp_ns=np.array([r['timestamp_ns'] for r in rows],np.int64),track_id=np.array([r['track_id'] for r in rows]),frame_index=np.array([r['frame_index'] for r in rows]),crop_rotations=rot,crop_focals=focal)
            del model
        seal=dict(complete=True,GT_inputs=False,files={p.name:sha(p) for p in folder.glob('*.npz')},roundtrip_max_px=parity,baseline_raw_vs_FK_projection_max_px=float(np.abs(roundtrip-old).max()),input_sha256={str(p.relative_to(view)):sha(p) for p in [view/'outputs/wilor_clip2102/observations.npz',view/'outputs/wilor_clip2102/observations.json']})
        (folder/'seal.json').write_text(json.dumps(seal,indent=2));seals[str((folder/'seal.json').relative_to(R))]=sha(folder/'seal.json');print(json.dumps(dict(clip=clip['clip'],complete=True,roundtrip_max_px=parity)),flush=True)
        del rgb;torch.cuda.empty_cache()
    (R/'observations_frozen.json').write_text(json.dumps(dict(complete=True,freeze_unix=time.time(),models=lock,seals=seals,hand_GT_read=False,source_sha256=sha(Path(__file__)),scope='All historical diagnostics; no fresh generalization claim'),indent=2))
if __name__=='__main__':main()
