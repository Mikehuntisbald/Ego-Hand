"""Trainable spatial stem on full native 16x12x1280 WiLoR features."""
import torch
from torch import nn
from spatial_rgb_model import SpatialHead
import spatial_rgb_common as s

class NativeSpatialStem(nn.Module):
    def __init__(self,device,bank=None):
        super().__init__();head=SpatialHead().to(device).eval();head.load_state_dict(torch.load(s.RUN/'sealed/rgb_probe.pt',weights_only=False,map_location=device)['model'])
        self.project=head.project;self.spatial=head.spatial;self.bank=bank
    def forward(self,x):
        z=self.project(x.float()).transpose(1,2).reshape(-1,128,16,12)
        return (z+self.spatial(z)).flatten(2).transpose(1,2)
    def replace_batch(self,b,data,ids):
        f=data['feature_ids'][ids];unique,inverse=torch.unique(f,return_inverse=True);keep=unique>0
        features=torch.zeros(len(unique),192,128,device=f.device,dtype=torch.float16)
        features[keep]=self(self.bank[unique[keep]]).half()
        b['rgb']=features[inverse.flatten()].reshape(len(ids),17,192,128)
        return b

def dense_bank(device,n):
    bank=torch.zeros(n,192,1280,device=device,dtype=torch.float16);last=0
    for path in sorted((s.RUN/'dense_chunks').glob('*.pt')):
        c=torch.load(path,weights_only=False,mmap=True);assert c['start']==last;last=c['end'];bank[c['start']+1:c['end']+1]=c['features'][:,0].to(device)
    assert last==n-1
    return bank
