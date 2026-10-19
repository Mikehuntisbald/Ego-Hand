"""Independent image localization; coarse projection enters risk heads only."""
import torch
from torch import nn
from torch.nn import functional as F
class RGBObservation(nn.Module):
    def __init__(self,width=128,mode='spatial'):
        super().__init__();assert mode in ['spatial','pooled'];self.mode=mode
        self.stem=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,width),nn.GELU())
        self.blocks=nn.Sequential(nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.GELU(),nn.Conv2d(width,width,3,padding=1),nn.GroupNorm(8,width),nn.GELU())
        self.upsample=nn.Sequential(nn.ConvTranspose2d(width,96,4,stride=2,padding=1),nn.GroupNorm(8,96),nn.GELU(),nn.ConvTranspose2d(96,64,4,stride=2,padding=1),nn.GroupNorm(8,64),nn.GELU(),nn.Conv2d(64,21,1))
        self.inband=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,21))
        self.risk=nn.Sequential(nn.Linear(width+42+42,width),nn.GELU(),nn.Linear(width,42))
        yy,xx=torch.meshgrid((torch.arange(64)+.5)/64,(torch.arange(48)+.5)/48,indexing='ij');self.register_buffer('grid',torch.stack([xx,yy],-1).reshape(1,1,-1,2))
    def forward(self,rgb,coarse_uv):
        x=self.stem(rgb.float());pooled=x.mean(1)
        if self.mode=='pooled':x=pooled[:,None].expand(-1,192,-1)
        x=x.transpose(1,2).reshape(len(x),-1,16,12);x=x+self.blocks(x);logits=self.upsample(x).float().flatten(2)
        logp=logits.log_softmax(-1);uv=(logp.exp()[...,None]*self.grid).sum(-2)
        risk=self.risk(torch.cat([pooled.float(),uv.detach().flatten(1),coarse_uv.float().clamp(-3,4).flatten(1)],1)).reshape(-1,2,21)
        return dict(uv=uv,logp=logp,inband_logit=self.inband(pooled).float(),coarse_bad_logit=risk[:,0].float(),new_bad_logit=risk[:,1].float())
def pixel_error(uv,target):return ((uv-target)*uv.new_tensor([192.,256.])).norm(dim=-1)
def loss(model,out,target):
    uv=target['uv'];mask=target['inband'];difference=(model.grid-uv[:,:,None])*uv.new_tensor([192.,256.]);gaussian=(-difference.square().sum(-1)/(2*4.**2)).softmax(-1)
    kl=(gaussian*(gaussian.clamp_min(1e-12).log()-out['logp'])).sum(-1);coordinate=((out['uv']-uv).abs().sum(-1)*mask).sum()/mask.sum().clamp_min(1)
    heat=(kl*mask).sum()/mask.sum().clamp_min(1);framing=F.binary_cross_entropy_with_logits(out['inband_logit'],mask.float())
    risk=F.binary_cross_entropy_with_logits(out['coarse_bad_logit'],(pixel_error(target['coarse_uv'],uv)>8).float())+F.binary_cross_entropy_with_logits(out['new_bad_logit'],(pixel_error(out['uv'].detach(),uv)>8).float())
    return heat+5*coordinate+.2*framing+.2*risk,dict(heatmap=float(heat.detach()),coordinate=float(coordinate.detach()),framing=float(framing.detach()),risk=float(risk.detach()))
def blend(out,coarse_uv):
    weight=.25*out['inband_logit'].sigmoid()*out['coarse_bad_logit'].sigmoid()*(1-out['new_bad_logit'].sigmoid())
    return coarse_uv+weight[...,None]*(out['uv']-coarse_uv),weight
