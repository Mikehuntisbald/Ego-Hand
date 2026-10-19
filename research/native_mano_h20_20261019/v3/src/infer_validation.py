"""Frozen native-head predictions on six clips; hand annotations are forbidden."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np,torch
from scipy.spatial.transform import Rotation
R=Path('/mnt/why/native_mano_multiseq_v3_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019')
sys.path.insert(0,str(V2/'src'))
from native_dit import NativeMANODiT
from shared_noise import SharedNoiseDiT
from native_mano import NativeMANO,exp_so3
from train_native import predict
from temporal_window_v44 import SHORT_OFFSETS

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class Observations:
    def __init__(self,view):
        coarse=np.load(view/'outputs/wilor_clip2102/observations.npz')
        self.rows=json.loads((view/'outputs/wilor_clip2102/observations.json').read_text())['frames'];self.N=len(self.rows)
        head=json.loads((view/'outputs/head_clip2102/headpose.json').read_text())['frames']
        self.state=torch.tensor(coarse['state'],device='cuda');self.beta=torch.tensor(coarse['betas'],device='cuda')
        self.right=torch.tensor(coarse['right'],device='cuda');self.rgb=torch.tensor(coarse['rgb_spatial'],device='cuda')
        self.scores=torch.tensor([x['confidence'] for x in self.rows],device='cuda')
        poses=np.array([head[x['frame_index']]['T_world_from_rgb'] for x in self.rows]);self.R=torch.tensor(poses[:,:3,:3],device='cuda',dtype=torch.float32);self.T=torch.tensor(poses[:,:3,3],device='cuda',dtype=torch.float32)
        windows=[];dt=[];valid=[]
        for i,row in enumerate(self.rows):
            ids=np.array([j for j,x in enumerate(self.rows) if x['track_id']==row['track_id'] and head[x['frame_index']]['segment']==head[row['frame_index']]['segment']]);stamps=np.array([self.rows[j]['timestamp_ns'] for j in ids],np.int64)
            offsets=np.array(SHORT_OFFSETS);targets=int(row['timestamp_ns'])+np.rint(offsets*1e9).astype(np.int64);distance=np.abs(targets[:,None]-stamps[None]);near=distance.argmin(1);good=distance[np.arange(17),near]<=20_000_000
            win=ids[near];win[8]=i;good[8]=True;windows.append(win);dt.append((stamps[near]-row['timestamp_ns'])/1e9);valid.append(good)
        self.windows=torch.tensor(np.array(windows),device='cuda');self.dt=torch.tensor(np.array(dt),device='cuda',dtype=torch.float32);self.valid=torch.tensor(np.array(valid),device='cuda')

    def batch(self,begin,end):
        ids=self.windows[begin:end];s=self.state[ids];r=self.R[ids];t=self.T[ids]
        world=torch.einsum('btij,btj->bti',r,s[...,:3])+t;relative=torch.einsum('bji,btj->bti',r[:,8],world-world[:,8:9])/.3
        globalR=torch.einsum('bji,btjk,btkl->btil',r[:,8],r,exp_so3(s[...,3:6]));local=exp_so3(s[...,6:].reshape(len(ids),17,15,3))
        six=lambda x:x[...,:,:2].transpose(-1,-2).flatten(-2)
        tokens=torch.cat([torch.cat([relative,torch.zeros_like(relative)],-1)[:,:,None],six(globalR)[:,:,None],six(local)],2)
        return dict(rgb=self.rgb[ids],coarse_tokens=tokens,valid=self.valid[begin:end],dt=self.dt[begin:end],scores=self.scores[ids],right_window=self.right[ids],state=self.state[begin:end],betas=self.beta[begin:end],right=self.right[begin:end])

def protected(codec,b,delta):
    original=codec.from_state(b['state'],b['betas'],b['right']);lo=torch.zeros(len(delta),device='cuda');hi=torch.ones_like(lo)
    for _ in range(16):
        mid=(lo+hi)/2;p=codec.apply_residual(b['state'],b['betas'],b['right'],delta*mid[:,None,None])['joints']
        distance=(p-original).norm(dim=-1)*1000
        good=(distance.square().mean(1).sqrt()<=5)&(torch.quantile(distance,.95,dim=1)<=10)&(distance.amax(1)<=30)
        lo=torch.where(good,mid,lo);hi=torch.where(good,hi,mid)
    return codec.apply_residual(b['state'],b['betas'],b['right'],delta*lo[:,None,None]),lo

def run_model(model,codec,data,guard=False,zero_rgb=False):
    parts={};max_parity=0.;all_delta=[];all_alpha=[]
    with torch.inference_mode():
        for begin in range(0,data.N,8):
            end=min(data.N,begin+8);b=data.batch(begin,end)
            with torch.autocast('cuda',dtype=torch.bfloat16):result,delta,_=predict(model,codec,b,zero_rgb=zero_rgb)
            alpha=torch.ones(len(delta),device='cuda')
            if guard:result,alpha=protected(codec,b,delta);delta=delta*alpha[:,None,None]
            native=codec.export_native(result);recovered=codec.decode_exported(native)
            max_parity=max(max_parity,float((recovered-result['joints']).abs().max()))
            assert torch.equal(result['wrist'],b['state'][:,:3]) and torch.equal(result['betas'],b['betas'])
            native['joints_camera21']=result['joints'];native['root_rotation_physical']=result['root_rotation']
            for name,value in native.items():parts.setdefault(name,[]).append(value.cpu().numpy())
            all_delta.append(delta.cpu().numpy());all_alpha.append(alpha.cpu().numpy())
    assert max_parity<1e-6
    result={k:np.concatenate(v) for k,v in parts.items()};result['normalized_residual']=np.concatenate(all_delta);result['protection_alpha']=np.concatenate(all_alpha)
    result['track_id']=np.array([r['track_id'] for r in data.rows]);result['timestamp_ns']=np.array([r['timestamp_ns'] for r in data.rows],np.int64)
    return result,max_parity

def forbid_hand_labels(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=str(args[0]);assert not any(x in name for x in ['.hands.json','__hand_shapes.json__','training_targets.pt','/targets.pt','evaluation_labels.pt']),f'Forbidden inference label read: {name}'

def main():
    torch.set_num_threads(4);sys.addaudithook(forbid_hand_labels)
    protocol=json.loads((R/'protocol.json').read_text());codec=NativeMANO();start=time.time();seals={}
    # Check the refactor against a sealed, previously computed v2 output.
    prior=Observations(V2/'coherent_frontend');record=protocol['models']['cosine_dit'];assert sha(Path(record['path']))==record['sha256']
    ck=torch.load(record['path'],weights_only=False,map_location='cuda');model=NativeMANODiT(**ck['config']).cuda().eval();model.load_state_dict(ck['model'])
    value,parity=run_model(model,codec,prior,guard=True);expected=np.load(V2/'coherent_frontend/outputs/native_cosine_dit/native_parameters.npz')
    diff=float(np.max(np.abs(value['joints_camera21']-expected['joints_camera21'])));assert diff<1e-6,diff
    (R/'receipts/inference_refactor_parity.json').write_text(json.dumps(dict(complete=True,prior_sealed_output_max_m=diff,native_redecode_max_m=parity,hand_labels_forbidden=True),indent=2))
    del prior,model;torch.cuda.empty_cache()
    # Checkpoint choices and sampling settings were locked before these labels.
    for clip in protocol['clips']:
        view=Path(clip['view']);data=Observations(view);folder=view/'predictions';folder.mkdir(exist_ok=False);reports={}
        with torch.inference_mode():xyz=codec.from_state(data.state,data.beta,data.right).cpu().numpy()
        np.savez_compressed(folder/'WiLoR.npz',joints_camera21=xyz,track_id=np.array([x['track_id'] for x in data.rows]),timestamp_ns=np.array([x['timestamp_ns'] for x in data.rows],np.int64))
        for arm,record in protocol['models'].items():
            assert sha(Path(record['path']))==record['sha256']
            ck=torch.load(record['path'],weights_only=False,map_location='cuda');model=NativeMANODiT(**ck['config']).cuda().eval();model.load_state_dict(ck['model'])
            output,parity=run_model(model,codec,data);np.savez_compressed(folder/f'{arm}.npz',**output);reports[arm]=dict(native_redecode_max_m=parity,wrist_unchanged=True,shape_unchanged=True)
            if arm=='cosine_dit':
                value,parity=run_model(model,codec,data,guard=True);np.savez_compressed(folder/'cosine_protected_displacement.npz',**value)
                value,parity=run_model(model,codec,data,zero_rgb=True);np.savez_compressed(folder/'cosine_RGB_off.npz',**value)
                shared=SharedNoiseDiT(**ck['config']).cuda().eval();shared.load_state_dict(ck['model']);value,parity=run_model(shared,codec,data);np.savez_compressed(folder/'cosine_shared.npz',**value);del shared
            del model
        files={p.name:sha(p) for p in folder.glob('*.npz')}
        payload=dict(complete=True,source_clip=clip['clip'],sequence=clip['sequence'],observations=data.N,files=files,geometry=reports,hand_GT_read=False,
            input_hashes={str(p.relative_to(view)):sha(p) for p in [view/'outputs/wilor_clip2102/observations.npz',view/'outputs/wilor_clip2102/observations.json',view/'outputs/head_clip2102/headpose.json']},
            camera_trajectory_source='provided dataset reference, not predicted headpose',no_BA=True,production_admitted=False)
        (folder/'seal.json').write_text(json.dumps(payload,indent=2));seals[str((folder/'seal.json').relative_to(R))]=sha(folder/'seal.json')
        print(json.dumps(dict(clip=clip['clip'],predictions_complete=True,observations=data.N,seconds=time.time()-start)),flush=True)
        del data;torch.cuda.empty_cache()
    dependencies=[V2/'src'/n for n in ['native_dit.py','native_mano.py','train_native.py','native_data.py','shared_noise.py']]
    seal=dict(complete=True,freeze_unix=time.time(),seals=seals,hand_GT_read_before_freeze=False,reference_camera_used=True,scope=protocol['scope'],protocol_sha256=sha(R/'protocol.json'),sources={p.name:sha(p) for p in (R/'src').glob('*.py')},dependencies={str(p):sha(p) for p in dependencies})
    (R/'receipts/predictions_frozen.json').write_text(json.dumps(seal,indent=2));print(json.dumps(dict(complete=True,seconds=time.time()-start)),flush=True)

if __name__=='__main__':main()
