"""Training-only MANO/UmeTrack wrist definition calibration.

Paired GT parameters are used ONLY to learn a static landmark convention,
never as inference pose/shape. Right-hand teacher samples; left mirrors it.
"""
import json,tarfile,time
from pathlib import Path
import torch,numpy as np
from hand3d_v8_common import V7,save
from hand_tracking_toolkit.hand_models.mano_hand_model import MANO_TO_CANONICAL_LANDMARK_MAPPING
from wilor.utils.geometry import aa_to_rotmat
import smplx

def main():
    torch.set_num_threads(4);device='cuda:0';out=V7.parent/'joint_mano_v29';out.mkdir(exist_ok=True)
    roles=json.loads((V7.parent/'dense_sampling_v13/source_roles.json').read_text());root=V7.parent.parent
    betas=[];thetas=[];wrists=[];gt=[];subjects=[];sequences=[]
    for r in roles:
        if r['role']!='train':continue
        split,seq,clip=r['split'],r['sequence'],r['clip']
        with tarfile.open(root/'rgb_clips'/split/seq/f'clip-{clip:06d}.tar') as archive:
            shape=json.load(archive.extractfile('__hand_shapes.json__'))['mano']
        lines=[json.loads(x) for x in (root/'export/annotations'/split/seq/f'clip-{clip:06d}.jsonl').read_text().splitlines()]
        for frame in lines[::10]:
            for hand in frame['hands']:
                if hand['side']!='right' or hand['mano_pose'] is None:continue
                betas.append(shape);thetas.append(hand['mano_pose']['thetas']);wrists.append(hand['mano_pose']['wrist_xform']);gt.append(hand['xyz_world_m']);subjects.append(r['subject']);sequences.append(seq)
    beta=torch.tensor(betas,device=device);theta=torch.tensor(thetas,device=device);wr=torch.tensor(wrists,device=device);gt=torch.tensor(gt,device=device)
    model=smplx.create(str(root/'experiments/yolo26_wilor_3d_20261003/assets/MANO_RIGHT.pkl'),'mano',use_pca=True,is_rhand=True,num_pca_comps=15).to(device)
    with torch.no_grad():
        pieces=[]
        for i in range(0,len(beta),256):
            x=model(betas=beta[i:i+256],hand_pose=theta[i:i+256],global_orient=wr[i:i+256,:3],transl=wr[i:i+256,3:])
            j=x.joints
            if j.shape[1]!=21:j=torch.cat([j,x.vertices[:,[744,320,443,554,671]]],1)
            pieces.append(j[:,MANO_TO_CANONICAL_LANDMARK_MAPPING])
        mano=torch.cat(pieces);R=aa_to_rotmat(wr[:,:3])
        target=torch.einsum('nj,njk->nk',gt[:,5]-mano[:,5],R)
    # Equal subject weights, avoiding many-frame subjects dominating calibration.
    unique=sorted(set(subjects));X=torch.stack([beta[torch.tensor([s==u for s in subjects],device=device)].mean(0) for u in unique]).double()
    Y=torch.stack([target[torch.tensor([s==u for s in subjects],device=device)].median(0).values for u in unique]).double()
    mean=X.mean(0);yc=Y.mean(0);xc=X-mean
    W=torch.linalg.solve(xc.T@xc+torch.eye(10,device=device,dtype=torch.float64)*.1,xc.T@(Y-yc))
    bias=yc-mean@W
    constant=yc.float();pred=bias.float()+beta@W.float()
    result=dict(wrist_bias_m=bias.float().cpu(),wrist_shape_m=W.float().cpu(),wrist_constant_m=constant.cpu(),training_subjects=unique)
    torch.save(result,out/'landmark_adapter.pt')
    save(out/'landmark_calibration.json',dict(complete=True,scope='Training-only static wrist landmark convention. GT shape/pose never used at inference; predictedbeta/globalrot only. Left reflects right convention; needs heldoutleft audit.',
        frames=len(beta),subjects=unique,constant_offset_m=constant.cpu().tolist(),bias_m=bias.cpu().tolist(),shape_matrix_m=W.cpu().tolist(),
        raw_wrist_offset_mean_mm=float(target.norm(dim=-1).mean()*1000),constant_residual_mean_mm=float((target-constant).norm(dim=-1).mean()*1000),
        shape_residual_mean_mm=float((target-pred).norm(dim=-1).mean()*1000),subject_median_offset_m={u:Y[i].cpu().tolist() for i,u in enumerate(unique)}))
    print((out/'landmark_calibration.json').read_text(),flush=True)

if __name__=='__main__':main()
