"""YOLO26s visual backbone + calibrated, uncertain canonical-20 3D pose head."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ultralytics import YOLO
from export_hand_labels import EDGES

class CoarsePose3D(nn.Module):
    def __init__(self,weight='/mnt/why/HOT3D/weights/yolo26s.pt'):
        super().__init__()
        full=YOLO(weight).model
        self.backbone=nn.ModuleList(list(full.model[:11]))
        self.low=nn.Conv2d(256,128,1);self.high=nn.Conv2d(512,128,1)
        self.geometry=nn.Sequential(nn.Linear(21,128),nn.SiLU(),nn.Linear(128,128))
        self.queries=nn.Parameter(torch.randn(1,20,128)*.02)
        self.position=nn.Linear(2,128)
        self.attn=nn.MultiheadAttention(128,4,batch_first=True)
        layer=nn.TransformerEncoderLayer(128,4,512,batch_first=True,norm_first=True,dropout=0)
        self.joints=nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.relative_head=nn.Linear(128,3);self.sigma_head=nn.Linear(128,1)
        self.root_head=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,4))
        nn.init.zeros_(self.relative_head.weight);nn.init.zeros_(self.relative_head.bias)
        nn.init.zeros_(self.sigma_head.weight);nn.init.zeros_(self.sigma_head.bias)
        nn.init.zeros_(self.root_head[-1].weight);nn.init.zeros_(self.root_head[-1].bias)
        self.root_head[-1].bias.data[2]=math.log(math.expm1(.35))
        for p in self.parameters():p.requires_grad_(True)
    def forward(self,image,geometry):
        x=image;low=None
        for i,layer in enumerate(self.backbone):
            assert layer.f==-1
            x=layer(x)
            if i==6:low=x
        maps=F.silu(self.low(low)+F.interpolate(self.high(x),size=low.shape[-2:],mode='bilinear',align_corners=False))
        h,w=maps.shape[-2:];yy,xx=torch.meshgrid((torch.arange(h,device=x.device)+.5)/h,(torch.arange(w,device=x.device)+.5)/w,indexing='ij')
        xy=torch.stack([xx,yy],dim=-1).reshape(-1,2)
        tokens=maps.flatten(2).transpose(1,2)
        positioned=tokens+self.position(2*xy-1)[None]
        g=self.geometry(geometry)
        q=self.queries.expand(x.shape[0],-1,-1)+g[:,None]
        attended,attention=self.attn(q,positioned,positioned,need_weights=True)
        joint=self.joints(q+attended)
        relative=.10*self.relative_head(joint);relative=relative-relative[:,5:6]
        root_raw=self.root_head(torch.cat([tokens.mean(dim=1),g],dim=-1))
        z=.15+F.softplus(root_raw[:,2:3]);ray=geometry[:,-3:]
        root_xy=ray[:,:2]/ray[:,2:3].clamp_min(.1)*z+.15*root_raw[:,:2]
        root=torch.cat([root_xy,z],dim=-1)
        log_sigma=self.sigma_head(joint).squeeze(-1).clamp(-3,2)
        root_log_sigma=root_raw[:,3].clamp(-3,2)
        uv=attention@xy
        cache=F.adaptive_avg_pool2d(maps,8).flatten(2).transpose(1,2)
        return dict(xyz=root[:,None]+relative,relative=relative,root=root,uv=uv,
                    log_sigma=log_sigma,root_log_sigma=root_log_sigma,
                    confidence=torch.exp(-.03*log_sigma.exp()/.025),
                    root_confidence=torch.exp(-.10*root_log_sigma.exp()/.05),rgb_tokens=cache)

def coarse_loss(pred,gt,uv_gt,uv_valid):
    root=gt[:,5];relative=gt-root[:,None]
    mask=torch.ones(20,device=gt.device);mask[5]=0
    rel_error=(pred['relative']-relative)/.03;root_error=(pred['root']-root)/.1
    relative_loss=(F.smooth_l1_loss(pred['relative']/.03,relative/.03,reduction='none').mean(-1)*mask).sum()/mask.sum()/gt.shape[0]
    root_loss=F.smooth_l1_loss(pred['root']/.1,root/.1)
    uv_loss=(F.smooth_l1_loss(pred['uv'],uv_gt,reduction='none').mean(-1)*uv_valid).sum()/uv_valid.sum().clamp_min(1)
    nll=((.5*rel_error.square().mean(-1)*torch.exp(-2*pred['log_sigma'])+pred['log_sigma'])*mask).sum()/mask.sum()/gt.shape[0]
    nll=nll+(.5*root_error.square().mean(-1)*torch.exp(-2*pred['root_log_sigma'])+pred['root_log_sigma']).mean()
    a,b=zip(*EDGES)
    gt_bones=(relative[:,a]-relative[:,b]).norm(dim=-1)
    bones=(pred['relative'][:,a]-pred['relative'][:,b]).norm(dim=-1)
    bone_loss=F.l1_loss(bones,gt_bones)/.03
    loss=relative_loss+root_loss+.5*uv_loss+.05*nll+.1*bone_loss
    return loss,dict(relative=relative_loss,root=root_loss,uv=uv_loss,nll=nll,bone=bone_loss)
