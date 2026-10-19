"""Audit actual right-hand MANO GT against the UmeTrack20 supervision proxy.

Post-freeze diagnostic only. Decode dataset MANO parameters with the licensed
existing RIGHT asset; never fabricate LEFT parameters from the RIGHT PCA.
"""
import hashlib,json,sys,tarfile
from pathlib import Path
import cv2,numpy as np,torch
from scipy.optimize import linear_sum_assignment
R=Path('/mnt/why/native_mano_rollout_v4_20261019');V=Path('/mnt/why/native_mano_multiseq_v3_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019')
sys.path.insert(0,str(V2/'src'));sys.path.insert(0,str(V/'src'));sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/src')
from native_mano import NativeMANO,exp_so3,common
from geometry import transform,camera
from evaluate_validation import iou,stats
import smplx
OPENPOSE=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]

def main():
    torch.set_num_threads(4);protocol=json.loads((V/'protocol.json').read_text());assert (V/'evaluation_r3/metrics.json').exists()
    right_asset=common.RUN/'assets/MANO_RIGHT.pkl';layer=smplx.create(str(right_asset),'mano',use_pca=True,is_rhand=True,num_pca_comps=15).eval();codec=NativeMANO(device='cpu')
    old_GT=np.load(V/'evaluation_r3/labels_and_errors.npz');records=[];all_pred={a:[] for a in ['WiLoR','regression','cosine_dit','cosine_protected_displacement','rollout_dit','rollout_protected','rollout_dit_displacement']};shape_pred=[]
    for clip in protocol['clips']:
        view=Path(clip['view']);rows=json.loads((view/'outputs/wilor_clip2102/observations.json').read_text())['frames'];coarse=np.load(view/'outputs/wilor_clip2102/observations.npz')
        predictions={a:np.load(view/('predictions_rollout' if a.startswith('rollout_') else 'predictions')/f'{a}.npz')['joints_camera21'] for a in all_pred}
        with tarfile.open(clip['archive']) as tar:
            beta=json.load(tar.extractfile('__hand_shapes.json__'))['mano']
            for fi in range(150):
                frame=f'{fi:06}';hands=json.load(tar.extractfile(frame+'.hands.json'));cal=json.load(tar.extractfile(frame+'.cameras.json'))['214-1'];pose=transform(cal['T_world_from_camera'])
                targets=[]
                for side,h in hands.items():
                    if not h.get('umetrack_pose') or '214-1' not in h.get('boxes_amodal',{}):continue
                    box=np.array(h['boxes_amodal']['214-1']);box[[0,2]]=np.clip(box[[0,2]],0,1408);box[[1,3]]=np.clip(box[[1,3]],0,1408)
                    if np.maximum(box[2:]-box[:2],0).prod()>=64:targets.append((side,box))
                ids=[i for i,r in enumerate(rows) if r['frame_index']==fi]
                if not ids or not targets:continue
                overlap=np.array([[iou(np.array(rows[i]['box_xyxy']),t[1]) for t in targets] for i in ids]);a,b=linear_sum_assignment(1-overlap)
                for x,y in zip(a,b):
                    side=targets[y][0]
                    if overlap[x,y]<.1 or side!='right' or not hands[side].get('mano_pose'):continue
                    i=ids[x];h=hands[side]['mano_pose'];records.append(dict(clip=clip['clip'],observation=i,image=rows[i]['image'],box=rows[i]['box_xyxy'],calibration=cal['calibration'],beta=beta,theta=h['thetas'],wrist_xform=h['wrist_xform'],R=pose[:3,:3],T=pose[:3,3],UM_camera=old_GT[f'clip{clip["clip"]}_GT_camera20'][i]))
                    for arm in all_pred:all_pred[arm].append(predictions[arm][i])
                    shape_pred.append(coarse['betas'][i])
    beta=torch.tensor(np.array([r['beta'] for r in records]),dtype=torch.float32);theta=torch.tensor(np.array([r['theta'] for r in records]),dtype=torch.float32);root=torch.tensor(np.array([r['wrist_xform'] for r in records]),dtype=torch.float32)
    with torch.no_grad():
        gt=layer(betas=beta,global_orient=root[:,:3],hand_pose=theta,transl=root[:,3:],return_verts=True,return_full_pose=True)
        full=gt.full_pose;assert full.shape==(len(records),48)
        rotations=exp_so3(full[:,:3]);local=exp_so3(full[:,3:].reshape(-1,15,3))
        xyz,vertices=codec.decode(gt.joints[:,0],rotations,local,beta,torch.ones(len(records),dtype=torch.bool),vertices=True)
        mesh_error=float((vertices-gt.vertices).abs().max());assert mesh_error<1e-6,mesh_error
        wrong_shape=codec.decode(gt.joints[:,0],rotations,local,torch.tensor(np.array(shape_pred),dtype=torch.float32),torch.ones(len(records),dtype=torch.bool))
    xyz=xyz.numpy();Rcam=np.array([r['R'] for r in records]);Tcam=np.array([r['T'] for r in records]);GT=np.einsum('nji,nkj->nki',Rcam,xyz-Tcam[:,None]);UM=np.array([r['UM_camera'] for r in records]);mapped=GT[:,common.MAPPING]
    rel=lambda x,w:x-x[:,w:w+1]
    cross=np.linalg.norm(mapped-UM,axis=-1)*1000;cross_rel=np.linalg.norm(rel(mapped,5)-rel(UM,5),axis=-1)*1000
    shape_error=np.linalg.norm(rel(wrong_shape.numpy(),0)-rel(xyz,0),axis=-1)*1000
    metrics={}
    for arm,value in all_pred.items():
        p=np.array(value);metrics[arm]=dict(camera21_mm=stats(np.linalg.norm(p-GT,axis=-1).ravel()*1000),relative20_excluding_wrist_mm=stats(np.linalg.norm(rel(p,0)-rel(GT,0),axis=-1)[:,1:].ravel()*1000),wrist_mm=stats(np.linalg.norm(p[:,0]-GT[:,0],axis=-1)*1000))
    result=dict(complete=True,posthoc_geometry_audit=True,hand_side='right only; no MANO_LEFT asset or fabricated left PCA',hand_frames=len(records),MANO_asset_sha256=hashlib.sha256(right_asset.read_bytes()).hexdigest(),official_GT_mesh_redecode_max_m=mesh_error,
        target='actual dataset MANO15 PCA + shape10 + world rotvec/transl, decoded through official SMPLX; native21 uses WiLoR exact fingertip/order convention',
        GT_MANO_vs_UM_mapped20=dict(camera_mm=stats(cross.ravel()),relative_without_wrist_mm=stats(cross_rel[:,np.arange(20)!=5].ravel()),wrist_mm=stats(cross[:,5])),
        GT_pose_with_predicted_shape_only_relative_error_mm=stats(shape_error[:,1:].ravel()),metrics=metrics,
        production_admitted=False,inference_or_training_changed=False,not_comparable_to_previous_full20_or_private_GT21_protocol=True)
    (R/'native_GT_right_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
