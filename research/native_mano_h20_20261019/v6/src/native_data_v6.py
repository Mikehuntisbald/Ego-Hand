"""Train-only counterfactual pose FK, with the input's fixed predicted shape."""
import sys
from pathlib import Path
import torch
sys.path.insert(0,'/mnt/why/native_mano_supervision_v5_20261019/src')
from native_data_v5 import Data as V5Data
ROOT=Path('/mnt/why/native_mano_shape_v6_20261019')
class Data(V5Data):
    def __init__(self,*args,supervision='native',**kwargs):
        assert supervision in ['native','compatible'];self.mode=supervision
        super().__init__(*args,supervision='native',**kwargs)

    def training_target(self,indices):return self.target(indices)

    @torch.no_grad()
    def pose_target(self,b,t,codec):
        if self.mode=='native':return t['gt']
        # All rotations here are loss-only native GT. Neither these nor GT beta
        # are injected into b or the network. Input/predicted beta stays fixed.
        return codec.decode(t['gt'][:,0],t['root_rotation'],t['local_rotation'],b['betas'],b['right'])
