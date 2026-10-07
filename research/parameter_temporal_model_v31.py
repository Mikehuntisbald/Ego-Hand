"""RGB-conditioned temporal hand-parameter generation with hard FK decoder."""
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_native_v10 import NativeTrajectoryHand3D
from parameter_codec_v31 import ParameterCodec

class ParameterTemporalHand(NativeTrajectoryHand3D):
    def __init__(self,kind='regression',device='cpu'):
        super().__init__(kind,True);self.codec=ParameterCodec(device)
        mask=torch.zeros(1,1,21,3);mask.flatten()[:34]=1;self.token_mask=mask
        self.parameter_position=nn.Parameter(torch.randn(1,1,21,self.width)*.02)
        self.side_head=nn.Sequential(nn.LayerNorm(self.width),nn.Linear(self.width,2));nn.init.zeros_(self.side_head[-1].weight);nn.init.zeros_(self.side_head[-1].bias)
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def load_visual_initial(self,state):
        own=self.state_dict();kept={k:v for k,v in state.items() if k in own and own[k].shape==v.shape and not k.startswith(('head.','noisy.')) and k!='token_mask'}
        self.load_state_dict(kept,strict=False)
        return len(kept)

    def encode(self,b):
        joint,memory,c,heat=super().encode(b)
        B=len(c);valid=b['rgb_valid'].float();h=joint.reshape(B,17,21,self.width).mean(2)
        context=c+(h*valid[:,:,None]).sum(1)/valid.sum(1,keepdim=True).clamp_min(1)
        logits=self.side_head(context)
        prior=F.one_hot(b['predicted_right'].long(),2).float()*4
        heat['side_logits']=logits+prior;return joint+self.parameter_position.expand(B,17,-1,-1).flatten(1,2),memory,c,heat

    def generate(self,b,seed,draws=4,steps=10):
        encoded=self.encode(b);delta=self.sample_trajectory(b,encoded,steps,draws,seed)
        state=b['kinematic_coarse'][None]+delta
        # Averaging parameter candidates preserves the hand-model decoding;
        # never average independently modified outputXYZ skeletons.
        mean=state.mean(0);side=encoded[3]['side_logits'].argmax(-1)
        xyz=self.codec.decode(mean,side)
        return dict(state=mean,xyz=xyz,right=side,side_logits=encoded[3]['side_logits'],encoded=encoded)

    def objective(self,b,target,seed):
        generated=self.generate(b,seed);state,xyz=generated['state'],generated['xyz'];gt=target['gt'];valid=target['valid'];vm=valid.float();vm[:,:,5]=0
        ce=(xyz-gt).norm(dim=-1);re=((xyz-xyz[:,:,5:6])-(gt-gt[:,:,5:6])).norm(dim=-1)
        fit=((ce/.03+.75*re/.03)*vm).sum()/vm.sum().clamp_min(1)
        desired=(target['state']-b['kinematic_coarse'])*self.token_mask
        pm=target['mask'][:,:,None,None]*self.token_mask
        if self.kind=='regression':
            parameter=(F.smooth_l1_loss(state,target['state'],reduction='none',beta=.5)*pm).sum()/pm.sum().clamp_min(1)
        else:
            t=torch.randint(100,(len(gt),),device=gt.device);a=self.alpha[t][:,None,None,None]
            noise=torch.randn_like(desired)*self.token_mask;signal=target['mask'][:,:,None,None]
            noisy=a.sqrt()*desired+(1-a).sqrt()*noise
            noisy=torch.where(signal,noisy,noise);v=self.net(noisy,t,generated['encoded'])
            expected=a.sqrt()*noise-(1-a).sqrt()*desired
            parameter=((v-expected).square()*pm).sum()/pm.sum().clamp_min(1)
        base=b['base'];g=gt[:,8];m=valid[:,8].clone();m[:,5]=False
        be=(base-g).norm(dim=-1);br=((base-base[:,5:6])-(g-g[:,5:6])).norm(dim=-1)
        goodc=m&(be<=.01);goodr=m&(br<=.01)
        protect=((ce[:,8]-be-.001).clamp_min(0)/.005*goodc).sum()/goodc.sum().clamp_min(1)+((re[:,8]-br-.001).clamp_min(0)/.005*goodr).sum()/goodr.sum().clamp_min(1)
        dt=b['dt'][:,1:]-b['dt'][:,:-1];mask=(valid[:,1:]&valid[:,:-1]&(dt>0)[:,:,None]).float()
        delta=(xyz[:,1:]-xyz[:,:-1])-(gt[:,1:]-gt[:,:-1])
        velocity=(F.smooth_l1_loss(delta/(dt.clamp_min(.001)[:,:,None,None]*.1),torch.zeros_like(delta),reduction='none',beta=.5).sum(-1)*mask).sum()/mask.sum().clamp_min(1)
        side=F.cross_entropy(generated['side_logits'],target['right'])
        shape=state.flatten(2)[:,:,29:34];consistency=((shape-shape[:,8:9]).square()*target['mask'][:,:,None]).sum()/target['mask'].sum().clamp_min(1)
        loss=fit+.3*parameter+2*protect+.05*velocity+.5*side+.02*consistency
        return loss,dict(fit=float(fit.detach()),parameter=float(parameter.detach()),protect=float(protect.detach()),velocity=float(velocity.detach()),side=float(side.detach()))

    @torch.inference_mode()
    def predict_parameters(self,b,seed=202610131):
        p=self.generate(b,seed)
        return dict(state=p['state'].float(),xyz_camera_m=p['xyz'][:,8].float(),trajectory_xyz_camera_m=p['xyz'].float(),right=p['right'],side_logits=p['side_logits'].float())
