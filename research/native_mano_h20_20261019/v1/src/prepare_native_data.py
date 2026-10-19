"""Natural RGB pilot; train-only fitted MANO pseudo-parameters, separate labels."""
import collections,hashlib,json,sys,time,tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
from native_mano import NativeMANO,common,exp_so3
from compare_detectors import iou
from prepare_joint_mano_v28 import records_bank
ROOT=Path('/mnt/why/native_mano_dit_20261019');OLD=Path('/mnt/why/HOT3D/experiments/online_rgb_3d_v47')
MAPPING=common.MAPPING

def save_json(path,d):path.write_text(json.dumps(d,indent=2))

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);folder=ROOT/'data';folder.mkdir(parents=True,exist_ok=True)
    assert not (folder/'ready.json').exists()
    d=torch.load(OLD/'inputs.pt',weights_only=False,mmap=True)
    groups=collections.defaultdict(list)
    for i,r in enumerate(d['rows']):groups[(d['roles'][i],d['subjects'][i],r['sequence'],r['clip'])].append(i)
    selected=[];selection=[]
    for subject in sorted({d['subjects'][i] for i,r in enumerate(d['roles']) if r=='train'}):
        key=max([k for k in groups if k[0]=='train' and k[1]==subject],key=lambda k:len(groups[k]))
        selected+=groups[key];selection.append(key)
    for role in ['dev_select','dev_calibrate']:
        key=max([k for k in groups if k[0]==role],key=lambda k:len(groups[k]));selected+=groups[key];selection.append(key)
    fids=d['feature_ids'][selected];unique=torch.unique(fids).tolist();unique=[i for i in unique if i>0]
    _,full=records_bank(d);assert len(full)==len(d['world'])-1
    K=len(unique)+1;remap=torch.zeros(len(d['world']),dtype=torch.long);remap[unique]=torch.arange(1,K)
    metadata=[];timestamp_cache={}
    def timestamp(rec):
        if 'timestamp_ns' in rec:return int(rec['timestamp_ns'])
        key=(rec['sequence'],int(rec['clip']))
        if key not in timestamp_cache:
            paths=list(Path('/mnt/why/HOT3D/rgb_clips').glob(f'*/{key[0]}/clip-{key[1]:06}.tar'));assert paths,key
            with tarfile.open(paths[0]) as tar:
                timestamp_cache[key]={int(m.name.split('.')[0]):int(json.load(tar.extractfile(m))['ref_timestamp_ns']) for m in tar.getmembers() if m.name.endswith('.info.json')}
        return timestamp_cache[key][int(rec['frame'])]
    for i in selected:
        r=dict(d['rows'][i]);r.update(role=d['roles'][i],subject=d['subjects'][i]);selected_fid=int(d['feature_ids'][i,8]);rec=full[selected_fid-1]
        metadata.append(dict(r,box=rec['box'],source_window_index=i,timestamp_ns=timestamp(rec)))
    save_json(folder/'selection.json',dict(groups=selection,source_indices=selected,windows=len(selected),observations=K-1,rule='largest window-count clip per train subject and per dev role; no label/error selection',natural_only=True))
    rgb=np.lib.format.open_memmap(folder/'rgb.npy',mode='w+',dtype=np.float16,shape=(K,192,1280));rgb[0]=0
    states=np.zeros((K,51),np.float32);betas=np.zeros((K,10),np.float32);rights=np.zeros(K,np.int64);scores=np.zeros(K,np.float32)
    matrices=np.tile(np.eye(3,dtype=np.float32),(K,1,1));translations=np.zeros((K,3),np.float32)
    model,_=common.load_model('cuda:0');decoder=NativeMANO();capture={}
    hook=model.backbone.register_forward_hook(lambda module,inp,out:capture.update(rgb=out[-1]))
    Q=np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32);start=time.time();maximum_raw_parity=0.
    def prep(fid):
        r=full[fid-1];side=r['side_predictions'];overlap=iou([r['box']],side['boxes'])[0]
        right=int(side['classes'][int(overlap.argmax())]) if len(overlap) and overlap.max()>=.05 else 1
        camera=r['camera']
        image=cv2.imread(r['image']);assert image is not None
        crop,R,f,error=common.crop(image,r['box'],camera,padding=1.3)
        return crop,R,f,right
    for begin in range(0,len(unique),16):
        ids=unique[begin:begin+16];n=len(ids)
        with ThreadPoolExecutor(max_workers=8) as pool:parts=list(pool.map(prep,ids))
        padded=parts+[parts[-1]]*(16-n);rs=np.array([p[3] for p in padded]);x=common.input_tensor([p[0] for p in padded],rs,1).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model({'img':x})
        params=out['pred_mano_params'];g=params['global_orient'].float().reshape(-1,3,3).cpu().numpy();local=params['hand_pose'].float().reshape(-1,15,3,3).cpu().numpy()
        beta=params['betas'].float().cpu().numpy();cam=out['pred_cam'].float().cpu().numpy();j=out['pred_keypoints_3d'].float().cpu().numpy()
        sign=rs*2-1;j[...,0]*=sign[:,None];cam[:,1]*=sign
        shift=np.stack([cam[:,1],cam[:,2],2*np.array([p[2] for p in padded])/(256*np.maximum(cam[:,0],1e-6))],-1)
        geom=np.array([p[1] for p in padded])@Q.T;S=np.tile(np.eye(3),(16,1,1));S[:,0,0]=sign
        xyz=np.einsum('nij,nkj->nki',geom,j+shift[:,None]);physical=geom@S@g@S
        state=np.concatenate([xyz[:,0],Rotation.from_matrix(physical).as_rotvec(),Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(-1,45)],-1).astype(np.float32)
        with torch.no_grad():reconstructed=decoder.from_state(torch.tensor(state,device='cuda'),torch.tensor(beta,device='cuda'),torch.tensor(rs,device='cuda')).cpu().numpy()
        maximum_raw_parity=max(maximum_raw_parity,float(np.linalg.norm(reconstructed-xyz,axis=-1).max()))
        ix=np.arange(begin+1,begin+n+1);states[ix]=state[:n];betas[ix]=beta[:n];rights[ix]=rs[:n]
        rgb[ix]=capture['rgb'][:n].flatten(2).transpose(1,2).float().cpu().numpy().astype(np.float16)
        matrices[ix]=d['rotation'][ids].numpy();translations[ix]=d['translation'][ids].numpy();scores[ix]=d['scores'][ids].numpy()
        if begin%160==0:print(json.dumps(dict(stage='native_WiLoR_RGB_cache',done=begin+n,total=K-1,seconds=time.time()-start)),flush=True)
    hook.remove();del model;rgb.flush();torch.cuda.empty_cache()
    pack=dict(state=torch.from_numpy(states),betas=torch.from_numpy(betas),right=torch.from_numpy(rights),rotation=torch.from_numpy(matrices),translation=torch.from_numpy(translations),
              scores=torch.from_numpy(scores),feature_ids=remap[fids],dt=d['dt'][selected].clone(),rows=metadata)
    torch.save(pack,folder/'inputs.pt')
    # Targets are read only for train/dev_select. No held-out parameter fitting.
    gt_raw=torch.load(OLD/'targets.pt',weights_only=False,mmap=True);target_indices=[i for i,r in enumerate(metadata) if r['role']!='dev_calibrate']
    center=pack['feature_ids'][target_indices,8];base=pack['state'][center].cuda();beta=pack['betas'][center].cuda();right=pack['right'][center].cuda()
    source=[selected[i] for i in target_indices];gt=gt_raw['gt'][source,8].cuda();valid=gt_raw['valid'][source,8].cuda()
    target_gt=gt-gt[:,5:6];delta=torch.nn.Parameter(torch.zeros(len(source),17,3,device='cuda'));opt=torch.optim.Adam([delta],lr=.04)
    for step in range(180):
        opt.zero_grad();normalized=delta.tanh()*torch.tensor([0.]+[1.]*16,device='cuda')[None,:,None]
        result=decoder.apply_residual(base,beta,right,normalized);j=result['joints'][:,MAPPING];relative=j-j[:,5:6]
        per=torch.nn.functional.smooth_l1_loss(relative/.02,target_gt/.02,reduction='none').sum(-1)
        loss=(per*valid).sum()/valid.sum().clamp_min(1)+.015*normalized.square().mean();loss.backward();opt.step()
    with torch.no_grad():
        target_delta=delta.tanh()*torch.tensor([0.]+[1.]*16,device='cuda')[None,:,None]
        fitted=decoder.apply_residual(base,beta,right,target_delta)['joints'][:,MAPPING];relative=fitted-fitted[:,5:6]
        fit_error=(relative-target_gt).norm(dim=-1)
        raw=decoder.from_state(base,beta,right)[:,MAPPING];raw_error=((raw-raw[:,5:6])-target_gt).norm(dim=-1)
    torch.save(dict(indices=torch.tensor(target_indices),delta=target_delta.detach().cpu(),gt=gt.cpu(),valid=valid.cpu(),
                    fit_error=fit_error.cpu(),coarse_relative_error=raw_error.cpu()),folder/'training_targets.pt')
    info=dict(complete=True,groups=selection,roles=dict(collections.Counter(r['role'] for r in metadata)),windows=len(selected),observations=K-1,
        native_parameter_source='actual WiLoR pred_mano_params; SO3 projection recorded',max_raw_BF16_vs_SO3_FK_mm=maximum_raw_parity*1000,
        RGB='full frozen WiLoR ViT 192x1280 spatial cells; real natural RGB; no synthetic occlusion',
        supervision='GT20 loss-only; train/dev_select fitted native MANO residual pseudo-targets, NOT ground-truth MANO rotation labels',
        heldout_target_fit=False,wrist_translation_learned=False,shape_learned=False,context='17 noncausal slots <=1.6s; upstream supplied camera trajectory used for temporal alignment',seconds=time.time()-start)
    save_json(folder/'ready.json',info);print(json.dumps(info),flush=True)

if __name__=='__main__':main()
