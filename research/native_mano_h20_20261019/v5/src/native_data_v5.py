"""Reuse only GT-free spatial inputs; separate actual native loss targets."""
from pathlib import Path
import sys,numpy as np,torch
ROOT=Path('/mnt/why/native_mano_supervision_v5_20261019')
OLD=Path('/mnt/why/native_mano_dit_20261019')
sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src')
from native_data import Data as BaseData

class Data(BaseData):
    def __init__(self,device='cuda:0',supervision='native'):
        self.supervision=supervision
        self.device=device;self.raw=torch.load(OLD/'data/inputs.pt',weights_only=False,map_location='cpu')
        self.rows=self.raw['rows'];self.bank={k:v.to(device) for k,v in self.raw.items() if torch.is_tensor(v)}
        self.rgb=torch.from_numpy(np.load(OLD/'data/rgb.npy')).to(device)
        self.labels=torch.load(ROOT/'data/native_targets.pt',weights_only=False,map_location=device)
        self.label_index={int(i):j for j,i in enumerate(self.labels['indices'])}
        self.train=[i for i in self.label_index if self.rows[i]['role']=='train']
        self.select=[i for i in self.label_index if self.rows[i]['role']=='dev_select']
        self.audit=[i for i,r in enumerate(self.rows) if r['role']=='dev_calibrate']
        if supervision=='proxy':
            self.proxy=torch.load(OLD/'data/training_targets.pt',weights_only=False,map_location=device)
            self.proxy_index={int(i):j for j,i in enumerate(self.proxy['indices'])}

    def training_target(self,indices):
        if self.supervision=='native':return self.target(indices)
        ix=torch.tensor([self.proxy_index[int(i)] for i in indices],device=self.device)
        return {k:v[ix] for k,v in self.proxy.items() if torch.is_tensor(v) and k!='indices'}
