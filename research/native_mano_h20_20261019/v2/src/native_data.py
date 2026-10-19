"""GT-free network inputs and separately addressed loss-only labels."""
from pathlib import Path
import numpy as np,torch
from native_mano import exp_so3
ROOT=Path('/mnt/why/native_mano_dit_20261019')

class Data:
    def __init__(self,device='cuda:0'):
        self.device=device;self.raw=torch.load(ROOT/'data/inputs.pt',weights_only=False,map_location='cpu')
        self.rows=self.raw['rows'];self.bank={k:v.to(device) for k,v in self.raw.items() if torch.is_tensor(v)}
        self.rgb=torch.from_numpy(np.load(ROOT/'data/rgb.npy')).to(device)
        self.labels=torch.load(ROOT/'data/training_targets.pt',weights_only=False,map_location=device)
        self.label_index={int(i):j for j,i in enumerate(self.labels['indices'])}
        self.train=[i for i,r in enumerate(self.rows) if r['role']=='train'];self.select=[i for i,r in enumerate(self.rows) if r['role']=='dev_select'];self.audit=[i for i,r in enumerate(self.rows) if r['role']=='dev_calibrate']
    def batch(self,indices,zero_rgb=False):
        ix=torch.as_tensor(indices,device=self.device,dtype=torch.long);b=self.bank;ids=b['feature_ids'][ix];valid=ids>0
        states=b['state'][ids];R=b['rotation'][ids];T=b['translation'][ids];root=states[...,:3]
        world=torch.einsum('btij,btj->bti',R,root)+T
        relative=torch.einsum('bji,btj->bti',R[:,8],world-world[:,8:9])/.3
        globalR=torch.einsum('bji,btjk,btkl->btil',R[:,8],R,exp_so3(states[...,3:6]))
        local=exp_so3(states[...,6:].reshape(*states.shape[:2],15,3))
        root_token=torch.cat([relative,torch.zeros_like(relative)],-1)[:,:,None]
        rot6=lambda x:x[...,:,:2].transpose(-1,-2).flatten(-2)
        tokens=torch.cat([root_token,rot6(globalR)[:,:,None],rot6(local)],2)
        rgb=self.rgb[ids];rgb=torch.zeros_like(rgb) if zero_rgb else rgb
        center=ids[:,8]
        return dict(rgb=rgb,coarse_tokens=tokens,valid=valid,dt=b['dt'][ix],scores=b['scores'][ids],right_window=b['right'][ids],
            state=b['state'][center],betas=b['betas'][center],right=b['right'][center],rotation=R[:,8],translation=T[:,8],indices=ix)
    def target(self,indices):
        assert all(int(i) in self.label_index for i in indices),'Held-out labels never enter training/selection'
        ix=torch.tensor([self.label_index[int(i)] for i in indices],device=self.device)
        return {k:v[ix] for k,v in self.labels.items() if torch.is_tensor(v) and k!='indices'}
