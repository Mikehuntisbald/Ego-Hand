"""Bounded hand parameters -> canonical20 current-eye XYZ.

Static shape PCA learned only from training hand models. GT never enters
codec inference. Parameters are decoded before any XYZ is returned.
"""
import dataclasses,math
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_v8_common import V7
from kinematic_hand_model_v30 import KinematicHandDecoder
from joint_mano_model_v29 import six_to_rotation,rotation_to_six
from hand_tracking_toolkit.hand_models.umetrack_hand_model import UmeTrackHandModelData

OUT=V7.parent/'parameter_kinematic_v31'

class ParameterCodec(KinematicHandDecoder):
    def __init__(self,device='cpu'):
        nn.Module.__init__(self)
        t=torch.load(OUT/'shape_space.pt',weights_only=False,map_location=device)
        for k,v in t.items():
            if torch.is_tensor(v):self.register_buffer(k,v.float())
        self.training_subjects=t['training_subjects'];self.shape_dimensions=5

    def shaped_model(self,beta,group):
        coef=beta[:,:5].clamp(-4,4)[group]
        vec=self.geometry_mean[None]+(coef*self.geometry_std[None])@self.geometry_basis
        n=len(group);j=vec[:,:66].reshape(n,22,3)*.05;l=vec[:,66:129].reshape(n,21,3)*.05
        axes=F.normalize(vec[:,129:].reshape(n,22,3)*.2,dim=-1)
        kwargs={}
        for field in dataclasses.fields(UmeTrackHandModelData):
            k=field.name
            if k in ['mesh_vertices','mesh_triangles','dense_bone_weights']:kwargs[k]=None
            elif k=='joint_rest_positions':kwargs[k]=j
            elif k=='landmark_rest_positions':kwargs[k]=l
            elif k=='joint_rotation_axes':kwargs[k]=axes
            else:v=getattr(self,k);kwargs[k]=v[None].expand(n,*v.shape)
        return UmeTrackHandModelData(**kwargs)

    def decode(self,state,right,shared=True):
        """State BxTx21x3. Shape shared over all T frames in each window."""
        B,T=state.shape[:2];flat=state.float().flatten(2)
        root=flat[:,:,:3].reshape(-1,3)*.1;rotation=flat[:,:,3:9].reshape(-1,6)
        normalized=flat[:,:,9:29].clamp(-.9999,.9999)
        angles=torch.logit((normalized+1)/2).reshape(-1,20)
        if shared:
            beta=flat[:,8,29:34].clamp(-4,4);group=torch.arange(B,device=state.device).repeat_interleave(T)
        else:
            beta=flat[:,:,29:34].reshape(-1,5).clamp(-4,4);group=torch.arange(B*T,device=state.device)
        sign=(1-2*right.float())[:,None].expand(-1,T).reshape(-1)
        with torch.autocast(device_type=state.device.type,enabled=False):
            x=self(angles,rotation,root,beta,group,sign)
        return x.reshape(B,T,20,3)

    @torch.no_grad()
    def coarse_states(self,xyz,right):
        """GT-free inverse-kinematic seeds from original WiLoR camera XYZ."""
        n=len(xyz);device=xyz.device;lo,hi=self.joint_limits[:20].unbind(-1)
        p=((self.neutral_angles[:20]-lo)/(hi-lo)).clamp(.001,.999)
        local=torch.logit(p)[None].expand(n,-1).clone()
        chains=[[5,6,7,0],[5,8,9,10,1],[5,11,12,13,2],[5,14,15,16,3],[5,17,18,19,4]]
        for f,c in enumerate(chains):
            for joint,points in [(4*f+3,c[-3:])]+([(4*f+2,c[1:4])] if f else []):
                a,b,z=points;bend=torch.acos(F.cosine_similarity(xyz[:,b]-xyz[:,a],xyz[:,z]-xyz[:,b],dim=-1).clamp(-.9999,.9999))
                q=((bend-lo[joint])/(hi[joint]-lo[joint])).clamp(.001,.999);local[:,joint]=torch.logit(q)
        beta=torch.zeros(n,5,device=device);group=torch.arange(n,device=device)
        canonical,_=self.canonical(local,beta,group);canonical-=canonical[:,5:6].clone()
        sign=1-2*right.float();canonical*=torch.stack([sign,torch.ones_like(sign),torch.ones_like(sign)],-1)[:,None]
        palm=[8,11,14,17];H=torch.einsum('nki,nkj->nij',canonical[:,palm],(xyz-xyz[:,5:6])[:,palm])
        U,_,V=torch.linalg.svd(H);V=V.transpose(-1,-2);D=torch.eye(3,device=device)[None].repeat(n,1,1);D[:,2,2]=torch.linalg.det(V@U.transpose(-1,-2))
        R=V@D@U.transpose(-1,-2);state=torch.zeros(n,21,3,device=device);v=state.flatten(1)
        v[:,:3]=xyz[:,5]/.1;v[:,3:9]=rotation_to_six(R);v[:,9:29]=2*self.angles(local)[:,:20].sub(lo)/(hi-lo)-1
        return state

def observation_batch(data,ids,prob,coarse_bank,right_bank,preserve_fitted=False):
    from hand3d_data_v7 import batch
    b=batch(data,ids,prob);f=data['feature_ids'][ids];center=f[:,8]
    x=coarse_bank[f].clone();v=x.flatten(2)
    rotation=six_to_rotation(v[:,:,3:9]);R=data['rotation'][center]
    aligned=R.transpose(-1,-2)[:,None]@data['rotation'][f]@rotation
    v[:,:,3:9]=rotation_to_six(aligned)
    if preserve_fitted:
        # Fitted roots differ from the source WiLoR wrist landmark. Transform
        # them using their own camera poses instead of replacing them with XYZ.
        worldroot=torch.einsum('btj,btkj->btk',v[:,:,:3]*.1,data['rotation'][f])+data['translation'][f]
        shifted=worldroot-data['translation'][center,None]
        v[:,:,:3]=torch.einsum('btj,bjk->btk',shifted,R)/.1
    else:
        v[:,:,:3]=b['xyz'][:,:,5]/.1
        v[:,:,29:34]=0.
    b['kinematic_coarse']=x;b['predicted_right']=right_bank[center]
    return b
