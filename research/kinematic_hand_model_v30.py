"""Training-template 20-landmark kinematics, bounded angles, shared shape.

Inference uses predicted boxes/XYZ/RGB only. No participant hand calibration.
"""
import math,dataclasses
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_v8_common import V7
from joint_mano_model_v29 import temporal_indices,rotation_to_six,six_to_rotation
from hand_tracking_toolkit.hand_models.umetrack_hand_model import UmeTrackHandModelData,UmeTrackHandPose,skin_landmarks,forward_kinematics

class KinematicHandDecoder(nn.Module):
    def __init__(self,device):
        super().__init__();t=torch.load(V7.parent/'joint_kinematic_v30/template.pt',weights_only=False,map_location=device)
        for k,v in t.items():
            if torch.is_tensor(v):self.register_buffer(k,v.float())
        self.training_subjects=t['training_subjects']

    def scales(self,beta):return .65+.85*beta.sigmoid()

    def angles(self,raw):
        lo,hi=self.joint_limits[:20].unbind(-1)
        return torch.cat([lo+(hi-lo)*raw.sigmoid(),torch.zeros(len(raw),2,device=raw.device)],-1)

    def shaped_model(self,beta,group):
        scale=self.scales(beta)[group];n=len(group)
        palm=scale[:,:1,None];joint=self.joint_rest_positions[None]*palm
        landmark=self.landmark_rest_positions[None]*palm
        fingers=[[0,6,7],[1,8,9,10],[2,11,12,13],[3,14,15,16],[4,17,18,19]]
        for f,lm in enumerate(fingers):
            base=self.joint_rest_positions[f*4][None]*palm[:,0]
            length=scale[:,f+1,None,None]
            joint[:,f*4:f*4+4]=base[:,None]+(self.joint_rest_positions[f*4:f*4+4][None]-self.joint_rest_positions[f*4][None,None])*length
            landmark[:,lm]=base[:,None]+(self.landmark_rest_positions[lm][None]-self.joint_rest_positions[f*4][None,None])*length
        kwargs={}
        for f in dataclasses.fields(UmeTrackHandModelData):
            if f.name in ['mesh_vertices','mesh_triangles','dense_bone_weights']:kwargs[f.name]=None
            elif f.name=='joint_rest_positions':kwargs[f.name]=joint
            elif f.name=='landmark_rest_positions':kwargs[f.name]=landmark
            else:
                value=getattr(self,f.name);kwargs[f.name]=value[None].expand(n,*value.shape)
        return UmeTrackHandModelData(**kwargs)

    def canonical(self,raw,beta,group):
        model=self.shaped_model(beta,group);n=len(raw)
        identity=torch.eye(4,device=raw.device)[None].expand(n,4,4)
        # The source skinning model carries an auxiliary landmark; official
        # exported annotation selects canonical indices0..19.
        local=skin_landmarks(model,self.angles(raw),identity)[:,:20]
        return local,model

    def forward(self,local,global_six,root,beta,group,sign):
        canonical,_=self.canonical(local,beta,group)
        relative=(canonical-canonical[:,5:6])*torch.stack([sign,torch.ones_like(sign),torch.ones_like(sign)],-1)[:,None]
        return torch.einsum('nij,nkj->nki',six_to_rotation(global_six),relative)+root[:,None]

    def initialize(self,obs,cache,rows):
        device=cache['base'].device;group,keys,pair,triple=temporal_indices(rows,device);n=len(rows)
        lo,hi=self.joint_limits[:20].unbind(-1)
        p=((self.neutral_angles[:20]-lo)/(hi-lo)).clamp(.001,.999)
        local=torch.logit(p)[None].expand(n,-1).clone()
        target=cache['baseline'];world=torch.einsum('njc,nkc->njk',target,cache['rotation'])+cache['translation'][:,None]
        # Anatomical finger bends from proposals give GT-free inverse-FK seeds.
        chains=[[5,6,7,0],[5,8,9,10,1],[5,11,12,13,2],[5,14,15,16,3],[5,17,18,19,4]]
        for f,chain in enumerate(chains):
            a,b,c=chain[-3:];bend=torch.acos(F.cosine_similarity(target[:,b]-target[:,a],target[:,c]-target[:,b],dim=-1).clamp(-.9999,.9999))
            j=4*f+3;q=((bend-lo[j])/(hi[j]-lo[j])).clamp(.001,.999);local[:,j]=torch.logit(q)
            if f:
                a,b,c=chain[1:4];bend=torch.acos(F.cosine_similarity(target[:,b]-target[:,a],target[:,c]-target[:,b],dim=-1).clamp(-.9999,.9999))
                j=4*f+2;q=((bend-lo[j])/(hi[j]-lo[j])).clamp(.001,.999);local[:,j]=torch.logit(q)
        beta=torch.full((len(keys),6),math.log((1-.65)/(1.5-1)),device=device)
        sign=1-2*obs['right'].float() # UmeTrack template is canonical LEFT.
        canonical,_=self.canonical(local,beta,group);canonical=canonical-canonical[:,5:6]
        canonical*=torch.stack([sign,torch.ones_like(sign),torch.ones_like(sign)],-1)[:,None]
        palm=[8,11,14,17];source=canonical[:,palm];target_world=(world-world[:,5:6])[:,palm]
        H=torch.einsum('nki,nkj->nij',source,target_world);U,_,V=torch.linalg.svd(H)
        V=V.transpose(-1,-2);D=torch.eye(3,device=device)[None].repeat(n,1,1);D[:,2,2]=torch.linalg.det(V@U.transpose(-1,-2))
        R=V@D@U.transpose(-1,-2)
        return dict(local=local,global_six=rotation_to_six(R),root=world[:,5],beta=beta,group=group,keys=keys,pair=pair,triple=triple,sign=sign)

    def pose_prior(self,local,initial):return ((self.angles(local)-self.angles(initial))/.5).square().mean()

    @torch.no_grad()
    def reconstruction_check(self,obs,cache,rows):
        initial=self.initialize(obs,cache,rows);group=initial['group'];n=len(rows)
        local,model=self.canonical(initial['local'],initial['beta'],group)
        R=six_to_rotation(initial['global_six']);reflection=torch.diag_embed(torch.stack([initial['sign'],torch.ones(n,device=R.device),torch.ones(n,device=R.device)],-1))
        T=torch.eye(4,device=R.device)[None].repeat(n,1,1);T[:,:3,:3]=R
        T[:,:3,3]=initial['root']-torch.einsum('nij,nj->ni',R@reflection,local[:,5])
        pose=UmeTrackHandPose(hand_side=obs['right'].long(),joint_angles=self.angles(initial['local']),wrist_xform=T)
        expected,_,_=forward_kinematics(pose,model,requires_mesh=False)
        actual=self(initial['local'],initial['global_six'],initial['root'],initial['beta'],group,initial['sign'])
        error=(actual-expected).norm(dim=-1)*1000
        return dict(passed=bool(error.max()<.02),official_fk_parity_mean_mm=float(error.mean()),official_fk_parity_max_mm=float(error.max()),
            shape='One palm/fivefinger scale vector shared over each predictedtrack',angles='20explicit bounded angles plus2fixedzero',
            training_subjects=self.training_subjects,subject_calibration_inference=False)
