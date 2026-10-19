"""Differentiable WiLoR-native MANO codec. No landmark IK in inference."""
import sys
from pathlib import Path
import torch
from torch import nn
REF=Path('/mnt/why/public_headhand_h20_20261019/vendor/Ego-Hand/research')
sys.path.insert(0,str(REF))
import wilor_eval_common as common

def exp_so3(v):
    x,y,z=v.unbind(-1);zero=torch.zeros_like(x)
    K=torch.stack([zero,-z,y,z,zero,-x,-y,x,zero],-1).reshape(*v.shape[:-1],3,3)
    theta=torch.linalg.vector_norm(v,dim=-1)
    A=torch.sinc(theta/torch.pi)[...,None,None]
    B=.5*torch.sinc(theta/(2*torch.pi)).square()[...,None,None]
    return torch.eye(3,device=v.device,dtype=v.dtype)+A*K+B*(K@K)

class NativeMANO(nn.Module):
    def __init__(self,device='cuda:0'):
        super().__init__()
        from wilor.models.mano_wrapper import MANO
        assets=common.RUN/'assets'
        self.layer=MANO(model_path=str(assets/'MANO_RIGHT.pkl'),data_dir=str(assets),gender='neutral',num_hand_joints=15,
            mean_params=str(assets/'mano_mean_params.npz'),create_body_pose=False).to(device).eval()
        self.layer.requires_grad_(False)
        self.register_buffer('scales',torch.tensor([0.,.2]+[.8]*15,device=device)[:,None])

    def decode(self,wrist,root_rotation,local_rotation,betas,right,vertices=False):
        with torch.autocast('cuda',enabled=False):
            return self._decode_fp32(wrist.float(),root_rotation.float(),local_rotation.float(),betas.float(),right,vertices)

    def _decode_fp32(self,wrist,root_rotation,local_rotation,betas,right,vertices=False):
        n=len(wrist);eye=torch.eye(3,device=wrist.device)[None,None].expand(n,1,3,3)
        out=self.layer(global_orient=eye,hand_pose=local_rotation,betas=betas,pose2rot=False)
        sign=torch.ones(n,3,device=wrist.device);sign[:,0]=right.to(wrist.dtype)*2-1
        local=(out.joints-out.joints[:,:1])*sign[:,None]
        xyz=torch.einsum('nij,nkj->nki',root_rotation,local)+wrist[:,None]
        if not vertices:return xyz
        mesh=torch.einsum('nij,nkj->nki',root_rotation,(out.vertices-out.joints[:,:1])*sign[:,None])+wrist[:,None]
        return xyz,mesh

    def from_state(self,state,betas,right,vertices=False):
        with torch.autocast('cuda',enabled=False):
            state=state.float()
            return self.decode(state[:,:3],exp_so3(state[:,3:6]),exp_so3(state[:,6:].reshape(-1,15,3)),betas.float(),right,vertices)

    def apply_residual(self,state,betas,right,normalized):
        with torch.autocast('cuda',enabled=False):
            return self._apply_fp32(state.float(),betas.float(),right,normalized.float())

    def _apply_fp32(self,state,betas,right,normalized):
        d=normalized.float()*self.scales
        root=exp_so3(state[:,3:6])@exp_so3(d[:,1])
        local=exp_so3(state[:,6:].reshape(-1,15,3))@exp_so3(d[:,2:])
        wrist=state[:,:3]+d[:,0]
        xyz=self.decode(wrist,root,local,betas,right)
        return dict(wrist=wrist,root_rotation=root,local_rotation=local,betas=betas,right=right,joints=xyz)

    def export_native(self,result):
        """Return actual MANO fields plus explicit WiLoR chirality restoration.

        S*(MANO(global_orient, hand_pose, betas, transl)) equals delivered joints.
        The exported transl is canonical MANO translation; wrist is separate.
        """
        with torch.autocast('cuda',enabled=False):
            return self._export_fp32(result)

    def _export_fp32(self,result):
        root,local,beta,right,wrist=[result[k] for k in ['root_rotation','local_rotation','betas','right','wrist']]
        sign=torch.ones(len(root),3,device=root.device);sign[:,0]=right.float()*2-1
        S=torch.diag_embed(sign);canonical_root=S@root@S
        identity=torch.eye(3,device=root.device)[None,None].expand(len(root),1,3,3)
        rest=self.layer(global_orient=identity,hand_pose=local,betas=beta,pose2rot=False).joints[:,0]
        # MANO LBS rotates about its regressed root: global rotation does not
        # rotate the nonzero template wrist around the model origin.
        transl=wrist*sign-rest
        return dict(global_orient=canonical_root[:,None],hand_pose=local,betas=beta,transl=transl,
                    reflection=sign,wrist_camera=wrist,root_rotation_physical=root)

    def decode_exported(self,p):
        with torch.autocast('cuda',enabled=False):
            out=self.layer(global_orient=p['global_orient'].float(),hand_pose=p['hand_pose'].float(),betas=p['betas'].float(),pose2rot=False)
            return (out.joints+p['transl'][:,None])*p['reflection'][:,None]
