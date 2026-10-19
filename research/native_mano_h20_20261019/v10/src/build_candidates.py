"""Freeze actual native DiT candidates and RGB unaries; hand GT forbidden."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation
R=Path('/mnt/why/native_candidate_sequence_v10_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019');V9=Path('/mnt/why/native_rgb_observation_v9_20261019');V3=Path('/mnt/why/native_mano_multiseq_v3_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019')
sys.path.insert(0,str(V2/'src'));sys.path.insert(0,str(V3/'src'));sys.path.insert(0,str(V9/'src'));sys.path.insert(0,str(V9/'eval_src'))
from native_mano import NativeMANO
from native_dit import NativeMANODiT
from infer_validation import Observations,protected
from infer_observations import crop_uv,forbid
from rgb_head import RGBObservation
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');a=ap.parse_args();torch.set_num_threads(4);sys.addaudithook(forbid)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3';assert int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())==0
    folder=R/('pilot' if a.pilot else 'candidates');folder.mkdir(parents=True,exist_ok=False)
    checkpoint=V7/'runs/full/best_development.pt';ck=torch.load(checkpoint,map_location='cuda',weights_only=False);model=NativeMANODiT(**ck['config']).cuda().eval();model.load_state_dict(ck['model']);codec=NativeMANO();p=V9/'runs/spatial/best_development.pt';imageck=torch.load(p,map_location='cpu',weights_only=False);image=RGBObservation(mode='spatial').cuda().eval();image.load_state_dict(imageck['model']);lock=dict(dit_checkpoint_sha256=sha(checkpoint),RGB_checkpoint_sha256=sha(p),protocol_sha256=sha(R/'protocol.json'),source_sha256=sha(Path(__file__)),GT_input=False)
    clips=json.loads((V3/'protocol.json').read_text())['clips']+[dict(clip=2102,view=str(V2/'coherent_frontend'))]
    if a.pilot:clips=clips[:1]
    seals={};started=time.time()
    for clip in clips:
        view=Path(clip['view']);data=Observations(view);coarse=np.load(view/'outputs/wilor_clip2102/observations.npz');head=json.loads((view/'outputs/head_clip2102/headpose.json').read_text())['frames'];N=min(data.N,24) if a.pilot else data.N;parts={};parity=0.;delta_parity=0.;export_parity=0.;refpath=V7/'full_pipeline/outputs/native_full/native_parameters.npz' if clip['clip']==2102 else V7/f"predictions/clip-{clip['clip']:06}/full_guard.npz";reference=np.load(refpath)
        with torch.inference_mode():
            for start in range(0,N,8):
                end=min(N,start+8);b=data.batch(start,end)
                with torch.autocast('cuda',dtype=torch.bfloat16):draws=model.sample(b,draws=4,steps=10,seed=190019)
                mean=draws.mean(0)*b['right'].float()[:,None,None];deltas=[mean,torch.zeros_like(mean)]+[d*b['right'].float()[:,None,None] for d in draws];native_parts={};joints=[];states=[];guarded=[];alphas=[]
                for k,delta in enumerate(deltas):
                    result,alpha=protected(codec,b,delta);delta=delta*alpha[:,None,None];native=codec.export_native(result);back=codec.decode_exported(native);export_parity=max(export_parity,float((back-result['joints']).abs().max()));assert torch.equal(result['wrist'],b['state'][:,:3]) and torch.equal(result['betas'],b['betas'])
                    if k==0:
                        parity=max(parity,float(np.abs(result['joints'].cpu().numpy()-reference['joints_camera21'][start:end]).max()));delta_parity=max(delta_parity,float(np.abs(delta.cpu().numpy()-reference['normalized_residual'][start:end]).max()))
                    for key,value in native.items():native_parts.setdefault(key,[]).append(value.cpu().numpy())
                    packed=np.concatenate([result['wrist'].cpu().numpy(),Rotation.from_matrix(result['root_rotation'].cpu().numpy()).as_rotvec(),Rotation.from_matrix(result['local_rotation'].cpu().numpy().reshape(-1,3,3)).as_rotvec().reshape(-1,45)],-1).astype(np.float32);states.append(packed);joints.append(result['joints'].cpu().numpy());guarded.append(delta.cpu().numpy());alphas.append(alpha.cpu().numpy())
                j=np.stack(joints,1);relative=j-j[:,:,:1];prior=.05*np.square((relative-relative[:,:1])[:,:,1:]/.005).mean(axis=(-1,-2));uv=np.stack([crop_uv(j[:,k],coarse['crop_rotations'][start:end],coarse['crop_focals'][start:end],coarse['right'][start:end].astype(bool)) for k in range(6)],1)
                basecrop=uv[:,1];condition=torch.tensor(basecrop,device='cuda');rgb=data.rgb[start:end]
                with torch.autocast('cuda',dtype=torch.bfloat16):observed=image(rgb,condition)
                weights=observed['inband_logit'].sigmoid()*observed['coarse_bad_logit'].sigmoid()*(1-observed['new_bad_logit'].sigmoid());inside=(condition>=0).all(-1)&(condition<=1).all(-1);weights*=inside*b['right'][:,None]
                probability=observed['logp'].exp().reshape(-1,21,64,48);grid=torch.tensor(uv.transpose(0,2,1,3)*2-1,device='cuda');sampled=F.grid_sample(probability.reshape(-1,1,64,48),grid.reshape(-1,6,1,2),mode='bilinear',padding_mode='zeros',align_corners=False).reshape(-1,21,6).transpose(1,2);nll=-sampled.clamp_min(1e-12).log();rgbcost=((nll-nll[:,:1])*weights[:,None]).mean(-1)
                values=dict(joints_camera21=j,state_camera51=np.stack(states,1),normalized_residual=np.stack(guarded,1),protection_alpha=np.stack(alphas,1),crop_uv=uv,RGB_cost=rgbcost.cpu().numpy(),prior_cost=prior,RGB_joint_weight=weights.cpu().numpy(),raw_draws=draws.cpu().numpy().transpose(1,0,2,3))
                values.update({key:np.stack(v,1) for key,v in native_parts.items()})
                for key,value in values.items():parts.setdefault(key,[]).append(value)
        assert parity<1e-6 and delta_parity<2e-5 and export_parity<1e-6,(parity,delta_parity,export_parity)
        value={k:np.concatenate(v) for k,v in parts.items()};rows=data.rows[:N];value.update(track_id=np.array([r['track_id'] for r in rows]),timestamp_ns=np.array([r['timestamp_ns'] for r in rows],np.int64),frame_index=np.array([r['frame_index'] for r in rows]),right=data.right[:N].cpu().numpy(),head_segment=np.array([head[r['frame_index']]['segment'] for r in rows]),camera_rotation=data.R[:N].cpu().numpy());out=folder/f"clip-{clip['clip']:06}";out.mkdir();target=out/'candidates.npz';np.savez_compressed(target,**value)
        displacement=np.linalg.norm(value['joints_camera21']-value['joints_camera21'][:,1:2],axis=-1)*1000;assert np.sqrt(np.mean(displacement**2,-1)).max()<=5.001 and np.quantile(displacement,.95,axis=-1).max()<=10.001 and displacement.max()<=30.001
        report=dict(complete=True,GT_input=False,observations=N,candidates=6,files={'candidates.npz':sha(target)},protected_mean_joint_parity_max_m=parity,protected_mean_delta_parity_max=delta_parity,native_export_parity_max_m=export_parity,reference_sha256=sha(refpath),protection_max_RMSE_mm=float(np.sqrt(np.mean(displacement**2,-1)).max()),protection_max_P95_mm=float(np.quantile(displacement,.95,axis=-1).max()),protection_max_point_mm=float(displacement.max()),source_inputs_sha256=sha(view/'outputs/wilor_clip2102/observations.npz'),RGB_unary_average_support=float(value['RGB_joint_weight'].mean()),inference_lock=lock)
        (out/'seal.json').write_text(json.dumps(report,indent=2));seals[str((out/'seal.json').relative_to(R))]=sha(out/'seal.json');print(json.dumps(dict(clip=clip['clip'],complete=True,seconds=time.time()-started,mean_parity=parity)),flush=True);del data;torch.cuda.empty_cache()
    (R/('pilot_frozen.json' if a.pilot else 'candidates_frozen.json')).write_text(json.dumps(dict(complete=True,freeze_unix=time.time(),seals=seals,GT_read=False,lock=lock,production_admitted=False),indent=2))
if __name__=='__main__':main()
