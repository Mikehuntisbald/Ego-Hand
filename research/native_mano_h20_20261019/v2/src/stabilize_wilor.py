"""Track evidence fixes crop chirality by actual WiLoR re-inference, never flag-only."""
import argparse,hashlib,json,sys
from pathlib import Path
import cv2,numpy as np,torch
from scipy.spatial.transform import Rotation
sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/src')
sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/vendor/Ego-Hand/research')
import wilor_eval_common as common
from geometry import load_frames,camera,IDENTITY
from mano_model import WristAnchoredMANO

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--public-root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4);cv2.setNumThreads(1)
    source=a.public_root/'outputs/wilor_clip2102';payload=json.loads((source/'observations.json').read_text());rows=payload['frames']
    coarse=np.load(source/'observations.npz');values={k:coarse[k].copy() for k in coarse.files};frames=load_frames(a.public_root/'data/clip-002102')
    decisions=[];changed=[]
    for tid in sorted({r['track_id'] for r in rows}):
        ids=[i for i,r in enumerate(rows) if r['track_id']==tid]
        votes=np.array([sum(rows[i]['side_confidence'] for i in ids if rows[i]['right']==s) for s in [0,1]])
        side=int(votes.argmax());margin=float(abs(votes[1]-votes[0])/max(votes.sum(),1e-8));support=sum(rows[i]['right']==side and rows[i]['side_confidence']>=.5 for i in ids)
        accepted=margin>=.6 and support>=3
        decisions.append(dict(track_id=tid,votes=votes.tolist(),margin=margin,strong_support=support,assigned_right=side if accepted else None))
        if accepted:
            for i in ids:
                if int(values['right'][i])!=side:changed.append((i,side))
    # No labels, handedness GT or pose errors participate in votes or crops.
    (a.output/'track_side_decisions.json').write_text(json.dumps(dict(GT_used=False,decisions=decisions,changed=changed),indent=2))
    model,_=common.load_model('cuda:0');decoder=WristAnchoredMANO(model.mano);capture={}
    hook=model.backbone.register_forward_hook(lambda m,inp,out:capture.update(rgb=out[-1]))
    Q=np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32);parity=[]
    for begin in range(0,len(changed),16):
        part=changed[begin:begin+16];crops=[];rotations=[];focals=[];flags=np.array([side for i,side in part])
        for i,side in part:
            row=rows[i];frame=frames[row['frame_index']];image=cv2.imread(frame['images']['214-1'])
            crop,R,f,_=common.crop(image,row['box_xyxy'],dict(calibration=frame['cameras']['214-1']['calibration'],T_world_from_camera=IDENTITY),padding=1.3)
            crops.append(crop);rotations.append(R);focals.append(f)
        inputs=common.input_tensor(crops,flags,1).cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):out=model({'img':inputs})
        p=out['pred_mano_params'];g=p['global_orient'].float().reshape(-1,3,3).cpu().numpy();local=p['hand_pose'].float().reshape(-1,15,3,3).cpu().numpy();beta=p['betas'].float().cpu().numpy()
        j=out['pred_keypoints_3d'].float().cpu().numpy();cam=out['pred_cam'].float().cpu().numpy();sign=flags*2-1;j[...,0]*=sign[:,None];cam[:,1]*=sign
        shift=np.stack([cam[:,1],cam[:,2],2*np.array(focals)/(256*np.maximum(cam[:,0],1e-6))],-1)
        geom=np.array(rotations)@Q.T;S=np.tile(np.eye(3),(len(part),1,1));S[:,0,0]=sign
        xyz=np.einsum('nij,nkj->nki',geom,j+shift[:,None]);physical=geom@S@g@S
        state=np.concatenate([xyz[:,0],Rotation.from_matrix(physical).as_rotvec(),Rotation.from_matrix(local.reshape(-1,3,3)).as_rotvec().reshape(-1,45)],-1).astype(np.float32)
        with torch.inference_mode():reconstructed=decoder(torch.tensor(state,device='cuda'),torch.tensor(beta,device='cuda'),torch.tensor(flags,device='cuda')).cpu().numpy()
        parity.append(float(np.linalg.norm(reconstructed-xyz,axis=-1).max()))
        rgb=capture['rgb'].flatten(2).transpose(1,2).float().cpu().numpy().astype(np.float16)
        for k,(i,side) in enumerate(part):
            row=rows[i];row['original_right']=row['right'];row['right']=side;row['side_source']='prediction-only confidence-weighted whole-track vote; actual crop rerun'
            row['side_review_required']=False;row['state_camera51']=state[k].tolist();row['betas10']=beta[k].tolist()
            row['xy']=camera(frames[row['frame_index']]['cameras']['214-1']['calibration']).eye_to_window(reconstructed[k]).tolist()
            for name,val in [('state',state),('betas',beta),('xyz_camera',reconstructed),('rgb_spatial',rgb),('crop_rotations',geom),('crop_focals',np.array(focals)),('right',flags)]:values[name][i]=val[k]
    hook.remove();same=np.ones(len(rows),bool);same[[i for i,s in changed]]=False
    assert all(np.array_equal(values[k][same],coarse[k][same]) for k in values),'Unchanged observations must remain exact'
    np.savez_compressed(a.output/'observations.npz',**values)
    payload.update(chirality_repair=dict(actual_WiLoR_reinference=True,flag_only_reassignment=False,changed=len(changed),unchanged_observations_exact=True,prediction_only=True,raw_SO3_parity_max_mm=max(parity,default=0)*1000))
    (a.output/'observations.json').write_text(json.dumps(payload,indent=2))
    seals={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file()}
    (a.output/'seal.json').write_text(json.dumps(dict(files=seals,GT_used=False),indent=2));print(json.dumps(payload['chirality_repair']),flush=True)

if __name__=='__main__':main()
