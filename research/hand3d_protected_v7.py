"""3D proposal training with learned trust and explicit correct-point protection."""
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_temporal_v7 import TemporalHand3D,WRIST,pack,unpack,EDGES

class ProtectedHand3D(TemporalHand3D):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.proposal_gate=nn.Sequential(nn.Linear(self.width+8,96),nn.SiLU(),nn.Linear(96,1))
        nn.init.zeros_(self.proposal_gate[-1].weight);nn.init.constant_(self.proposal_gate[-1].bias,-2)

    def gate_logits(self,delta,b,encoded):
        camera=torch.cat([b['risk_camera'][:,WRIST:WRIST+1],b['risk_camera']],1)
        relative=torch.cat([b['risk_camera'][:,WRIST:WRIST+1],b['risk_relative']],1)
        features=torch.cat([encoded[0],delta,pack(b['base']).clamp(-10,10),camera[...,None],relative[...,None]],-1)
        return self.proposal_gate(features)[...,0]

    def loss(self,b,gt,valid,gt_uv,uv_valid):
        tv=torch.cat([valid[:,WRIST:WRIST+1],valid],1).float()*self.token_mask[...,0]
        residual=(pack(gt)-pack(b['base']))*self.token_mask
        cam_before=(gt-b['base']).norm(dim=-1);rel_before=((gt-gt[:,WRIST:WRIST+1])-(b['base']-b['base'][:,WRIST:WRIST+1])).norm(dim=-1)
        weights=tv*(1+3*torch.cat([cam_before[:,WRIST:WRIST+1]>.02,rel_before>.02],1).float());encoded=self.encode(b)
        if self.kind=='dit':
            t=torch.randint(100,(len(gt),),device=gt.device);a=self.alpha[t][:,None,None];noise=torch.randn_like(residual)*self.token_mask
            x=a.sqrt()*residual+(1-a).sqrt()*noise;v=self.net(x,t,encoded);desired=a.sqrt()*noise-(1-a).sqrt()*residual
            clean=a.sqrt()*x-(1-a).sqrt()*v;error=(v-desired).square().sum(-1)+.3*(clean-residual).square().sum(-1)*(.1+self.alpha[t][:,None]);usable=(self.alpha[t]>.1).float()
        else:
            clean=self.net(torch.zeros_like(residual),torch.zeros(len(gt),dtype=torch.long,device=gt.device),encoded)
            error=F.smooth_l1_loss(clean,residual,reduction='none',beta=.5).sum(-1);usable=torch.ones(len(gt),device=gt.device)
        clean=clean.clamp(-8,8)*self.token_mask;logits=self.gate_logits(clean,b,encoded);gate=logits.sigmoid()*self.token_mask[...,0]
        prediction=unpack(pack(b['base'])+clean*gate[...,None])
        cam_after=(prediction-gt).norm(dim=-1);rel_after=((prediction-prediction[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1)
        geometry=((cam_after/.03+.5*rel_after/.03)*valid*usable[:,None]).sum()/(valid*usable[:,None]).sum().clamp_min(1)
        cg=valid&(cam_before<=.01);rg=valid&(rel_before<=.01);rg[:,WRIST]=False
        protect=((cam_after-cam_before-.001).clamp_min(0)/.005*cg*usable[:,None]).sum()/(cg*usable[:,None]).sum().clamp_min(1)
        protect+=((rel_after-rel_before-.001).clamp_min(0)/.005*rg*usable[:,None]).sum()/(rg*usable[:,None]).sum().clamp_min(1)
        # Training-only labels assess each proposed direction, not mere input presence.
        root_delta=clean[:,:1]*.1;pose_delta=clean[:,1:]*.03
        root_only=b['base']+root_delta;root_error=(root_only-gt).norm(dim=-1)
        good_harmed=(((root_error-cam_before)>.002)&cg).any(1)
        root_benefit=((cam_before-root_error)*valid).sum(1)/valid.sum(1).clamp_min(1)>.001
        root_target=root_benefit&~good_harmed
        pose_only=b['base']+pose_delta;pose_camera=(pose_only-gt).norm(dim=-1)
        pose_relative=((pose_only-pose_only[:,WRIST:WRIST+1])-(gt-gt[:,WRIST:WRIST+1])).norm(dim=-1)
        pose_target=(cam_before-pose_camera>.001)&(rel_before-pose_relative>.001)
        labels=torch.cat([root_target[:,None],pose_target],1).float().detach()
        gate_weights=tv*usable[:,None];gate_loss=(F.binary_cross_entropy_with_logits(logits,labels,reduction='none')*gate_weights).sum()/gate_weights.sum().clamp_min(1)
        bone_loss=prediction.sum()*0
        for u,v in EDGES:
            m=(valid[:,u]&valid[:,v]).float()*usable
            err=((prediction[:,u]-prediction[:,v]).norm(dim=-1)-(gt[:,u]-gt[:,v]).norm(dim=-1)).abs()/.01
            bone_loss+=(err*m).sum()/m.sum().clamp_min(1)/len(EDGES)
        loss=(error*weights).sum()/weights.sum().clamp_min(1)+.3*geometry+.5*protect+.2*gate_loss+.025*bone_loss
        if self.use_rgb:loss+=.025*self.visual_head.loss(encoded[3],gt_uv,uv_valid,b['positions'][:,8],b['roi'][:,8])
        return loss

    @torch.no_grad()
    def predict(self,b,steps=10,samples=4,seed=202610071):
        encoded=self.encode(b);B=len(b['base']);device=b['base'].device;gen=torch.Generator(device=device).manual_seed(seed);draws=[];gates=[]
        for si in range(samples if self.kind=='dit' else 1):
            if self.kind=='regression':delta=self.net(torch.zeros(B,21,3,device=device),torch.zeros(B,dtype=torch.long,device=device),encoded)
            else:
                x=torch.randn(B,21,3,device=device,generator=gen)*self.token_mask;schedule=torch.linspace(99,0,steps,device=device).round().long()
                for k,t in enumerate(schedule):
                    v=self.net(x,t.expand(B),encoded);a=self.alpha[t];clean=(a.sqrt()*x-(1-a).sqrt()*v).clamp(-8,8)*self.token_mask;eps=(1-a).sqrt()*x+a.sqrt()*v
                    x=clean if k+1==len(schedule) else self.alpha[schedule[k+1]].sqrt()*clean+(1-self.alpha[schedule[k+1]]).sqrt()*eps
                delta=x
            gate=self.gate_logits(delta,b,encoded).sigmoid()*self.token_mask[...,0];xyz=unpack(pack(b['base'])+delta*gate[...,None]);xyz=torch.where(b['confirmed'][...,None],b['base'],xyz);draws.append(xyz);gates.append(gate)
        draw=torch.stack(draws);return dict(xyz_camera_m=draw.mean(0),std_m=draw.std(0,unbiased=False),draws=draw,learned_gates=torch.stack(gates).mean(0))
