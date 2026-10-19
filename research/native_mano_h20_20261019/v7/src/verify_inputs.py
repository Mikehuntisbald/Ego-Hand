"""Sampled original-pixel/model parity plus full input-layout verification."""
import hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
R=Path('/mnt/why/native_mano_pool_v7_20261019')
sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src')
from native_mano import NativeMANO,common
from compare_detectors import iou
from prepare_joint_mano_v28 import records_bank
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    ready=json.loads((R/'data/ready.json').read_text());assert ready['complete']
    assert not (R/'data/native_targets.pt').exists(),'Input verification precedes native supervision'
    for name,v in ready['assets'].items():assert sha(R/'data'/name)==v['sha256'],name
    pack=torch.load(R/'data/inputs.pt',map_location='cpu',weights_only=False);old=torch.load('/mnt/why/HOT3D/experiments/online_rgb_3d_v47/inputs.pt',map_location='cpu',mmap=True,weights_only=False)
    protocol=json.loads((R/'protocol.json').read_text());selection=protocol['source_indices'];fids=old['feature_ids'][selection];unique=sorted(set(fids.flatten().tolist())-{0});mapping=torch.zeros(len(old['world']),dtype=torch.long);mapping[unique]=torch.arange(1,len(unique)+1)
    assert torch.equal(pack['feature_ids'],mapping[fids]) and torch.equal(pack['dt'],old['dt'][selection])
    for k in ['scores','rotation','translation']:assert torch.equal(pack[k][1:],old[k][unique]),k
    assert not any('gt' in k.lower() or 'target' in k.lower() for k in pack)
    assert all(r['role']=='train' and r['subject'] in protocol['subjects'] or r['role']=='dev_select' and r['subject']=='P0003' for r in pack['rows'])
    _,records=records_bank(old);crops=np.load(R/'data/crops.npy',mmap_mode='r');features=np.load(R/'data/rgb.npy',mmap_mode='r')
    rng=np.random.default_rng(190020);sample=np.unique(np.r_[np.linspace(1,len(unique),32).round().astype(int),rng.choice(np.arange(1,len(unique)+1),32,replace=False)]);sample=sample[:64]
    model,_=common.load_model('cuda:0');codec=NativeMANO();capture={};hook=model.backbone.register_forward_hook(lambda mod,inp,out:capture.update(features=out[-1]))
    max_feature=0.;max_beta=0.;max_wrist=0.;max_rotation=0.;pixel_checks=0
    Q=np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32)
    for begin in range(0,len(sample),16):
        chosen=sample[begin:begin+16];n=len(chosen);parts=[]
        for fid in chosen:
            record=records[unique[fid-1]-1];photo=cv2.imread(record['image']);assert photo is not None
            crop,rot,focal,_=common.crop(photo,record['box'],record['camera'],padding=1.3);assert np.array_equal(crop,crops[fid]);pixel_checks+=1
            side=record['side_predictions'];overlap=iou([record['box']],side['boxes'])[0];right=int(side['classes'][int(overlap.argmax())]) if len(overlap) and overlap.max()>=.05 else 1
            assert right==int(pack['right'][fid]);parts.append((rot,focal))
        padded=np.r_[chosen,np.repeat(chosen[-1],16-n)];rights=pack['right'][padded].numpy();inp=common.input_tensor([crops[i] for i in padded],rights,1).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model({'img':inp})
        replay=capture['features'][:n].flatten(2).transpose(1,2).float().cpu().numpy().astype(np.float16);delta=np.abs(replay.astype(np.float32)-features[chosen].astype(np.float32));max_feature=max(max_feature,float(delta.max()));assert np.array_equal(replay,features[chosen]),max_feature
        parameters=out['pred_mano_params'];beta=parameters['betas'][:n].float().cpu().numpy();max_beta=max(max_beta,float(np.abs(beta-pack['betas'][chosen].numpy()).max()));assert max_beta<1e-5
        g=parameters['global_orient'][:n].float().reshape(-1,3,3).cpu().numpy();local=parameters['hand_pose'][:n].float().reshape(-1,15,3,3).cpu().numpy();cam=out['pred_cam'][:n].float().cpu().numpy();xyz=out['pred_keypoints_3d'][:n].float().cpu().numpy();sign=rights[:n]*2-1;xyz[...,0]*=sign[:,None];cam[:,1]*=sign
        shift=np.stack([cam[:,1],cam[:,2],2*np.array([x[1] for x in parts])/(256*np.maximum(cam[:,0],1e-6))],-1);geom=np.array([x[0] for x in parts])@Q.T
        xyz=np.einsum('nij,nkj->nki',geom,xyz+shift[:,None]);max_wrist=max(max_wrist,float(np.abs(xyz[:,0]-pack['state'][chosen,:3].numpy()).max()));assert max_wrist<1e-6
        S=np.tile(np.eye(3),(n,1,1));S[:,0,0]=sign;root=geom@S@g@S
        cached=Rotation.from_rotvec(pack['state'][chosen,3:6].numpy()).as_matrix();physical=Rotation.from_matrix(root).as_matrix();max_rotation=max(max_rotation,float(np.abs(cached-physical).max()))
        cached_local=Rotation.from_rotvec(pack['state'][chosen,6:].numpy().reshape(-1,3)).as_matrix();local=Rotation.from_matrix(local.reshape(-1,3,3)).as_matrix();max_rotation=max(max_rotation,float(np.abs(cached_local-local).max()));assert max_rotation<1e-5
    hook.remove()
    parity=0.
    with torch.inference_mode():
        for begin in range(1,len(unique)+1,256):
            state=pack['state'][begin:begin+256].cuda();beta=pack['betas'][begin:begin+256].cuda();right=pack['right'][begin:begin+256].cuda();zero=torch.zeros(len(state),17,3,device='cuda');value=codec.apply_residual(state,beta,right,zero)
            assert torch.isfinite(value['joints']).all();difference=float((codec.decode_exported(codec.export_native(value))-value['joints']).abs().max());parity=max(parity,difference);assert difference<1e-6
    report=dict(complete=True,full_input_layout_and_source_metadata_equal=True,all_asset_hashes_verified=True,sampled_original_crop_byte_checks=pixel_checks,sampled_frozen_spatial_features_exact=True,sampled_feature_max_abs=max_feature,sampled_beta_max_abs=max_beta,sampled_wrist_max_m=max_wrist,sampled_rotation_matrix_max_abs=max_rotation,
        all_native_FK_states_checked=len(unique),native_export_redecode_max_m=parity,hand_annotation_or_target_files_opened=False,GT_used_in_inputs=False,
        scope='Original source pixel/model checks sample64; hashes/layout and native FK check all observations. Legacy data.pt read for index mapping only; no hand labels used as conditioning.',source_sha256=sha(Path(__file__)),production_admitted=False)
    (R/'receipts/input_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
