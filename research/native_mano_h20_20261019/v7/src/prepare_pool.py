"""Resumable natural-RGB native input cache for all original training clips.

No hand annotations or training targets are read. Preserve v1's predicted
side/crop contract; future small/full data controls use this same cache.
"""
import argparse,hashlib,json,os,sys,tarfile,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
R=Path('/mnt/why/native_mano_pool_v7_20261019');OLD=Path('/mnt/why/HOT3D/experiments/online_rgb_3d_v47');V1=Path('/mnt/why/native_mano_dit_20261019')
sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src')
from native_mano import NativeMANO,common
from compare_detectors import iou
from prepare_joint_mano_v28 import records_bank
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()
def write(p,x):
    temp=p.with_suffix('.partial.json');temp.write_text(json.dumps(x,indent=2));temp.replace(p)
def forbid(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=str(args[0]);assert not any(k in name for k in ['.hands.json','__hand_shapes.json__','training_targets.pt','native_targets.pt','/targets.pt']),name
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    torch.set_num_threads(4);cv2.setNumThreads(1);sys.addaudithook(forbid)
    folder=R/'data';folder.mkdir(parents=True,exist_ok=True);(R/'receipts').mkdir(exist_ok=True)
    assert not (folder/'ready.json').exists(),'Completed cache immutable'
    d=torch.load(OLD/'inputs.pt',weights_only=False,map_location='cpu',mmap=True);previous=torch.load(V1/'data/inputs.pt',weights_only=False,map_location='cpu')
    train=[i for i,r in enumerate(d['roles']) if r=='train'];select=[r['source_window_index'] for r in previous['rows'] if r['role']=='dev_select'];indices=train+select
    assert len(train)==2526 and len(select)==52
    train_people=sorted({d['subjects'][i] for i in train});assert train_people==['P0001','P0002','P0009','P0011','P0012','P0014']
    fids=d['feature_ids'][indices];unique=sorted(set(fids.flatten().tolist())-{0});K=len(unique)+1
    assert bool((fids[:,8]>0).all());assert float(d['dt'][indices].abs()[fids>0].max())<1.621
    _,full=records_bank(d);assert len(full)==len(d['world'])-1
    remap=torch.zeros(len(d['world']),dtype=torch.long);remap[unique]=torch.arange(1,K)
    timestamp_cache={}
    def timestamp(rec):
        if 'timestamp_ns' in rec:return int(rec['timestamp_ns'])
        key=(rec['sequence'],int(rec['clip']))
        if key not in timestamp_cache:
            paths=list(Path('/mnt/why/HOT3D/rgb_clips').glob(f'*/{key[0]}/clip-{key[1]:06}.tar'));assert len(paths)==1,key
            with tarfile.open(paths[0]) as tar:timestamp_cache[key]={int(m.name.split('.')[0]):int(json.load(tar.extractfile(m))['ref_timestamp_ns']) for m in tar.getmembers() if m.name.endswith('.info.json')}
        return timestamp_cache[key][int(rec['frame'])]
    rows=[]
    for i in indices:
        center=full[int(d['feature_ids'][i,8])-1];r=dict(d['rows'][i]);r.update(role=d['roles'][i],subject=d['subjects'][i],source_window_index=i,box=center['box'],timestamp_ns=timestamp(center));rows.append(r)
        assert all(full[int(fid)-1]['subject']==r['subject'] and full[int(fid)-1]['sequence']==r['sequence'] for fid in d['feature_ids'][i] if fid>0),'Context crossed subject/recording'
    selection=dict(source_indices=indices,small_source_indices=[r['source_window_index'] for r in previous['rows'] if r['role']=='train'],train_windows=len(train),dev_select_windows=len(select),
        training_clips=len({(d['rows'][i]['sequence'],d['rows'][i]['clip']) for i in train}),subjects=train_people,observations=K-1,
        input_source_sha256=sha(OLD/'inputs.pt'),model_sha256=sha(common.RUN/'assets/wilor_final.mirror.ckpt'),MANO_sha256=sha(common.RUN/'assets/MANO_RIGHT.pkl'),
        source_sha256=sha(Path(__file__)),contract='same predicted per-frame side/IoU rule, padding1.3 and rotation1 as native v1; natural unmodified RGB; no GT crop/side/pose/shape',
        context='original17 slots <=1.6s valid offsets; same subject and recording',pixel_cache='exact uint8 local BGR crop before rotation/flip/normalization; reconstruct using pinned common.input_tensor',
        hand_annotation_or_target_files_opened=False,GT_used_in_inputs=False,
        legacy_index_scope='records_bank reads legacy data.pt only for feature-ID/metadata mapping; no legacy pose labels are used to form RGB/coarse state/conditioning',
        camera_trajectory_scope='provided camera trajectory from the original input cache, same as current training; hand annotations are not conditioning')
    selection['geometry_and_pixel_sources']={str(p):sha(p) for p in [Path(common.__file__),Path('/mnt/why/native_mano_dit_v2_20261019/src/native_mano.py'),common.RUN/'assets/model_config.yaml',common.RUN/'assets/mano_mean_params.npz']}
    signature=hashlib.sha256(json.dumps(selection,sort_keys=True).encode()).hexdigest()
    if (R/'protocol.json').exists():assert json.loads((R/'protocol.json').read_text())['signature']==signature,'Existing input protocol changed'
    else:write(R/'protocol.json',dict(selection,signature=signature))
    progress=folder/'progress.json'
    if a.resume:
        saved=json.loads(progress.read_text());assert saved['signature']==signature;done=int(saved['completed'])
    else:assert not progress.exists();done=0
    specs={'rgb':((K,192,1280),np.float16),'crops':((K,256,256,3),np.uint8),'state':((K,51),np.float32),'betas':((K,10),np.float32),'right':((K,),np.int64),'scores':((K,),np.float32),'rotation':((K,3,3),np.float32),'translation':((K,3),np.float32)}
    bank={}
    for name,(shape,dtype) in specs.items():
        path=folder/f'{name}.npy';bank[name]=np.lib.format.open_memmap(path,mode='r+' if a.resume else 'w+',dtype=dtype,shape=shape)
        assert bank[name].shape==shape and bank[name].dtype==np.dtype(dtype)
        if not a.resume:bank[name][0]=0
    if not a.resume:
        bank['rotation'][0]=np.eye(3,dtype=np.float32)
        for v in bank.values():v.flush()
        write(progress,dict(signature=signature,completed=0,total=K-1,stage='input_only'))
    model,_=common.load_model('cuda:0');codec=NativeMANO();capture={};hook=model.backbone.register_forward_hook(lambda module,inp,out:capture.update(rgb=out[-1]))
    Q=np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32);start=time.time();maximum_raw_parity=float(saved.get('max_raw_parity_m',0)) if a.resume else 0.
    def prep(fid):
        r=full[fid-1];side=r['side_predictions'];overlap=iou([r['box']],side['boxes'])[0]
        right=int(side['classes'][int(overlap.argmax())]) if len(overlap) and overlap.max()>=.05 else 1
        image=cv2.imread(r['image']);assert image is not None,r['image']
        crop,rotation,focal,_=common.crop(image,r['box'],r['camera'],padding=1.3);assert crop.shape==(256,256,3) and crop.dtype==np.uint8
        return crop,rotation,focal,right
    with ThreadPoolExecutor(max_workers=8) as pool:
        for begin in range(done,len(unique),16):
            original_ids=unique[begin:begin+16];n=len(original_ids);parts=list(pool.map(prep,original_ids));padded=parts+[parts[-1]]*(16-n);rights=np.array([p[3] for p in padded]);inp=common.input_tensor([p[0] for p in padded],rights,1).cuda()
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):output=model({'img':inp})
            feature=capture['rgb'][:n].flatten(2).transpose(1,2).float().cpu().numpy().astype(np.float16)
            params=output['pred_mano_params'];g=params['global_orient'].float().reshape(-1,3,3).cpu().numpy();local=params['hand_pose'].float().reshape(-1,15,3,3).cpu().numpy();beta=params['betas'].float().cpu().numpy()
            cam=output['pred_cam'].float().cpu().numpy();xyz=output['pred_keypoints_3d'].float().cpu().numpy();sign=rights*2-1;xyz[...,0]*=sign[:,None];cam[:,1]*=sign
            shift=np.stack([cam[:,1],cam[:,2],2*np.array([p[2] for p in padded])/(256*np.maximum(cam[:,0],1e-6))],-1);geom=np.array([p[1] for p in padded])@Q.T
            xyz=np.einsum('nij,nkj->nki',geom,xyz+shift[:,None]);S=np.tile(np.eye(3),(16,1,1));S[:,0,0]=sign;physical=geom@S@g@S
            states=np.concatenate([xyz[:,0],Rotation.from_matrix(physical).as_rotvec(),Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(-1,45)],-1).astype(np.float32)
            with torch.no_grad():recovered=codec.from_state(torch.tensor(states,device='cuda'),torch.tensor(beta,device='cuda'),torch.tensor(rights,device='cuda')).cpu().numpy()
            maximum_raw_parity=max(maximum_raw_parity,float(np.linalg.norm(recovered-xyz,axis=-1).max()))
            ix=np.arange(begin+1,begin+n+1);bank['rgb'][ix]=feature;bank['crops'][ix]=np.stack([p[0] for p in parts]);bank['state'][ix]=states[:n];bank['betas'][ix]=beta[:n];bank['right'][ix]=rights[:n]
            bank['scores'][ix]=d['scores'][original_ids].numpy();bank['rotation'][ix]=d['rotation'][original_ids].numpy();bank['translation'][ix]=d['translation'][original_ids].numpy()
            if begin==done:
                replay=common.input_tensor([bank['crops'][i] for i in ix],bank['right'][ix],1);assert torch.equal(replay,inp[:n].cpu()),'Stored pixels change actual WiLoR input'
            completed=begin+n
            if completed%160==0 or completed==len(unique):
                for v in bank.values():v.flush()
                write(progress,dict(signature=signature,completed=completed,total=K-1,seconds=time.time()-start,max_raw_parity_m=maximum_raw_parity,GT_read=False));print(json.dumps(dict(stage='native_pixels_features',done=completed,total=K-1,seconds=time.time()-start)),flush=True)
    hook.remove();torch.cuda.empty_cache()
    pack={name:torch.from_numpy(np.array(bank[name])) for name in ['state','betas','right','scores','rotation','translation']};pack.update(feature_ids=remap[fids],dt=d['dt'][indices].clone(),rows=rows)
    temp=folder/'inputs.partial.pt';torch.save(pack,temp);temp.replace(folder/'inputs.pt')
    write(folder/'selection.json',selection);assets={name:dict(bytes=(folder/name).stat().st_size,sha256=sha(folder/name)) for name in ['inputs.pt','rgb.npy','crops.npy']}
    ready=dict(complete=True,train_windows=len(train),dev_select_windows=len(select),observations=K-1,training_clips=selection['training_clips'],full_spatial_RGB=True,native_MANO_states=True,
        exact_pixel_roundtrip=True,hand_annotation_or_target_files_opened=False,GT_used_in_inputs=False,model_and_assets_unchanged=True,maximum_raw_BF16_vs_SO3_FK_mm=maximum_raw_parity*1000,assets=assets,source_signature=signature,
        supervision_prepared=False,training_started=False,production_admitted=False)
    write(folder/'ready.json',ready);print(json.dumps(ready),flush=True)
if __name__=='__main__':main()
