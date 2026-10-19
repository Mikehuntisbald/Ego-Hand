"""Prediction-only input wrapper; labels are deliberately not a member."""
import sys
from pathlib import Path
import numpy as np,torch
V7=Path('/mnt/why/native_mano_pool_v7_20261019')
sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src')
from native_data import Data as Original
class Inputs(Original):
    def __init__(self,device='cuda:0'):
        self.device=device;self.raw=torch.load(V7/'data/inputs.pt',map_location='cpu',weights_only=False);self.rows=self.raw['rows'];self.bank={k:v.to(device) for k,v in self.raw.items() if torch.is_tensor(v)};self.rgb=torch.from_numpy(np.load(V7/'data/rgb.npy')).to(device)
