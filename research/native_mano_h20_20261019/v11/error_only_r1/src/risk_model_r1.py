"""Temporal RGB native3D quality. Error estimates never move individual joints."""
import torch
from torch import nn
from torch.nn import functional as F
class QualityBlock(nn.Module):
    def __init__(self,width):
        super().__init__();self.norm=nn.LayerNorm(width);self.cross=nn.MultiheadAttention(width,4,batch_first=True);self.ff=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width*4),nn.GELU(),nn.Linear(width*4,width))
    def forward(self,q,memory,padding):
        query=self.norm(q);q=q+self.cross(query,memory,memory,key_padding_mask=padding,need_weights=False)[0];return q+self.ff(q)
class NativeQuality(nn.Module):
    def __init__(self,width=96,use_rgb=True):
        super().__init__();self.use_rgb=use_rgb;self.rgb=nn.Sequential(nn.LayerNorm(1280),nn.Linear(1280,width),nn.GELU());self.patch_pos=nn.Parameter(torch.randn(1,1,192,width)*.02);self.parameter_pos=nn.Parameter(torch.randn(1,1,17,width)*.02);self.joint_pos=nn.Parameter(torch.randn(1,1,20,width)*.02)
        self.coarse=nn.Linear(6,width);self.frame=nn.Sequential(nn.Linear(5,width),nn.GELU(),nn.Linear(width,width));self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(width,4,width*4,dropout=0,batch_first=True,norm_first=True),1,enable_nested_tensor=False)
        self.point=nn.Sequential(nn.Linear(11,width),nn.GELU(),nn.Linear(width,width));self.proposal=nn.Linear(51,width);self.shape=nn.Linear(10,width);self.blocks=nn.ModuleList([QualityBlock(width) for _ in range(2)]);self.norm=nn.LayerNorm(width);self.delta=nn.Linear(width,1);self.harm=nn.Linear(width,1);self.good=nn.Linear(width,1)
        nn.init.zeros_(self.delta.weight);nn.init.zeros_(self.delta.bias);nn.init.constant_(self.harm.bias,-3.)
    def forward(self,b,c):
        image=b['rgb'].float();image=torch.zeros_like(image) if not self.use_rgb else image;B,T=image.shape[:2];info=torch.stack([b['dt'],b['dt'].abs(),b['scores'],b['right_window'].float(),b['valid'].float()],-1);frame=self.frame(info)
        rgb=self.rgb(image)+self.patch_pos+frame[:,:,None];coarse=self.coarse(b['coarse_tokens'])+self.parameter_pos+frame[:,:,None];padding=~b['valid'].bool();coarse=self.temporal(coarse.permute(0,2,1,3).reshape(B*17,T,-1),src_key_padding_mask=padding[:,None].expand(-1,17,-1).reshape(B*17,T));coarse=coarse.reshape(B,17,T,-1).permute(0,2,1,3)
        memory=torch.cat([coarse.flatten(1,2),rgb.flatten(1,2)],1);pad=torch.cat([padding[:,:,None].expand(-1,-1,17).flatten(1),padding[:,:,None].expand(-1,-1,192).flatten(1)],1)
        p=c['joints_camera21'].float();p=p-p[:,:,:1];reference=p[:,:1];change=p-reference;uv=(c['crop_uv'].float()[:,:,1:]-.5).mul(2).clamp(-4,4);point=torch.cat([p[:,:,1:]/.1,reference[:,:,1:].expand_as(p[:,:,1:])/.1,change[:,:,1:]/.005,uv],-1)
        q=self.point(point)+self.joint_pos+self.proposal(c['normalized_residual'].float().flatten(-2))[:,:,None]+self.shape(b['betas'].float())[:,None,None];K=q.shape[1];q=q.flatten(1,2)
        for block in self.blocks:q=block(q,memory,pad)
        q=self.norm(q).reshape(B,K,20,-1);distance=change[:,:,1:].norm(dim=-1)*1000;delta=distance*self.delta(q).squeeze(-1).tanh();move=distance>1.;harm_logit=self.harm(q).squeeze(-1);probability=harm_logit.sigmoid()*move;score=delta.mean(-1)/5;assert score.shape==(B,K)
        return dict(score=score,delta_error_mm=delta,harm_probability=probability,harm_logit=harm_logit,move=move,reference_good_logit=self.good(q[:,0]).squeeze(-1))
def objective(out,target,extra_preservation=False):
    delta=target['delta_error_mm'];harm=target['harm_on_good'].float();good=target['reference_good'].float();reg=F.smooth_l1_loss(out['delta_error_mm']/5,delta/5);active=out['move'];risk=(F.binary_cross_entropy_with_logits(out['harm_logit'],harm,reduction='none')*active).sum()/active.sum().clamp_min(1);accurate=F.binary_cross_entropy_with_logits(out['reference_good_logit'],good)
    gt_score=delta.mean(-1)/5;probability=(-gt_score/.15).softmax(-1);rank=(probability*(probability.clamp_min(1e-12).log()-(-out['score']/.15).log_softmax(-1))).sum(-1).mean();preserve=((-out['score']/.15).softmax(-1)*harm.mean(-1)).sum(-1).mean();total=reg+.2*risk+.1*accurate+.25*rank+(preserve if extra_preservation else 0.)
    return total,dict(delta_reg=float(reg.detach()),harm_BCE=float(risk.detach()),good_BCE=float(accurate.detach()),ranking=float(rank.detach()),expected_harm=float(preserve.detach()))
