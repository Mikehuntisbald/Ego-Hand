"""Same native model inputs, either the original small or expanded train set."""
import json,sys
from pathlib import Path
import numpy as np,torch
ROOT=Path('/mnt/why/native_mano_pool_v7_20261019')
sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src')
from native_data import Data as BaseData
class Data(BaseData):
    def __init__(self,device='cuda:0',data_arm='full',supervision='native'):
        assert data_arm in ['small','full'] and supervision=='native'
        self.device=device;self.raw=torch.load(ROOT/'data/inputs.pt',weights_only=False,map_location='cpu');self.rows=self.raw['rows']
        self.bank={k:v.to(device) for k,v in self.raw.items() if torch.is_tensor(v)}
        self.rgb=torch.from_numpy(np.load(ROOT/'data/rgb.npy')).to(device)
        labels=torch.load(ROOT/'data/native_targets.pt',weights_only=False,map_location='cpu');small=set(json.loads((ROOT/'data/selection.json').read_text())['small_source_indices'])
        keep=[j for j,i in enumerate(labels['indices']) if self.rows[int(i)]['role']=='dev_select' or self.rows[int(i)]['role']=='train' and (data_arm=='full' or self.rows[int(i)]['source_window_index'] in small)]
        self.labels={k:v[keep].to(device) for k,v in labels.items() if torch.is_tensor(v)}
        self.label_index={int(i):j for j,i in enumerate(self.labels['indices'])}
        self.train=[i for i in self.label_index if self.rows[i]['role']=='train'];self.select=[i for i in self.label_index if self.rows[i]['role']=='dev_select'];self.audit=[]
        assert len(self.select)==25 and len(self.train)==(155 if data_arm=='small' else 1330)
        assert sorted({self.rows[i]['subject'] for i in self.train})==['P0001','P0002','P0009','P0011','P0012','P0014']
        self.data_arm=data_arm
    def training_target(self,indices):return self.target(indices)
