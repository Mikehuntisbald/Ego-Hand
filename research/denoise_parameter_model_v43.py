"""Single-noise-timestep v-prediction with the same geometry auxiliaries.

Only the training objective changes: geometry is decoded from a denoised
training sample instead of differentiating a four-draw DDIM ensemble. The
inference network and ten-step/four-draw sampler remain identical.
"""
import torch
from torch.nn import functional as F
from semantic_parameter_model_v36 import SemanticParameterHand


class DenoiseParameterHand(SemanticParameterHand):
    def objective(self,b,target,seed):
        encoded=self.encode(b);desired=(target['state']-b['kinematic_coarse'])*self.token_mask
        t=torch.randint(100,(len(desired),),device=desired.device);a=self.alpha[t][:,None,None,None]
        noise=torch.randn_like(desired)*self.token_mask;signal=target['mask'][:,:,None,None]
        noisy=a.sqrt()*desired+(1-a).sqrt()*noise;noisy=torch.where(signal,noisy,noise)
        velocity_target=a.sqrt()*noise-(1-a).sqrt()*desired;velocity=self.net(noisy,t,encoded)
        pm=signal*self.token_mask
        parameter=((velocity-velocity_target).square()*pm).sum()/pm.sum().clamp_min(1)
        residual=(a.sqrt()*noisy-(1-a).sqrt()*velocity).clamp(-8,8)*self.token_mask
        state=b['kinematic_coarse']+residual;side=encoded[3]['side_logits'].argmax(-1)
        xyz=self.codec.decode(state,side);gt=target['gt'];valid=target['valid'];vm=valid.float();vm[:,:,5]=0
        ce=(xyz-gt).norm(dim=-1);re=((xyz-xyz[:,:,5:6])-(gt-gt[:,:,5:6])).norm(dim=-1)
        fit=((ce/.03+.75*re/.03)*vm).sum()/vm.sum().clamp_min(1)
        base=b['base'];g=gt[:,8];m=valid[:,8].clone();m[:,5]=False
        be=(base-g).norm(dim=-1);br=((base-base[:,5:6])-(g-g[:,5:6])).norm(dim=-1)
        goodc=m&(be<=.01);goodr=m&(br<=.01)
        protect=((ce[:,8]-be-.001).clamp_min(0)/.005*goodc).sum()/goodc.sum().clamp_min(1)
        protect+=((re[:,8]-br-.001).clamp_min(0)/.005*goodr).sum()/goodr.sum().clamp_min(1)
        dt=b['dt'][:,1:]-b['dt'][:,:-1];mask=(valid[:,1:]&valid[:,:-1]&(dt>0)[:,:,None]).float()
        delta=(xyz[:,1:]-xyz[:,:-1])-(gt[:,1:]-gt[:,:-1])
        motion=(F.smooth_l1_loss(delta/(dt.clamp_min(.001)[:,:,None,None]*.1),torch.zeros_like(delta),reduction='none',beta=.5).sum(-1)*mask).sum()/mask.sum().clamp_min(1)
        side_loss=F.cross_entropy(encoded[3]['side_logits'],target['right'])
        shape=state.flatten(2)[:,:,29:34]
        consistency=((shape-shape[:,8:9]).square()*target['mask'][:,:,None]).sum()/target['mask'].sum().clamp_min(1)
        loss=fit+.3*parameter+2*protect+.05*motion+.5*side_loss+.02*consistency
        return loss,dict(fit=float(fit.detach()),parameter=float(parameter.detach()),protect=float(protect.detach()),velocity=float(motion.detach()),side=float(side_loss.detach()))
