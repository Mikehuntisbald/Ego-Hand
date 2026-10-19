import json
from pathlib import Path
import torch
from input_data import Inputs
R=Path('/mnt/why/native_candidate_risk_v11_20261019')
class TrainingData(Inputs):
    def __init__(self,device='cuda:0'):
        assert json.loads((R/'data/inputs_frozen.json').read_text())['complete'];assert json.loads((R/'data/targets_ready.json').read_text())['complete'];super().__init__(device)
        bank=torch.load(R/'data/candidates.pt',map_location='cpu',weights_only=False);geometry=torch.load(R/'data/geometry.pt',map_location='cpu',weights_only=False);assert torch.equal(bank['indices'],geometry['indices']);target=torch.load(R/'data/quality_targets.pt',map_location='cpu',weights_only=False);ix=target['input_index'];self.indices=target['indices'].tolist();self.proposals={k:bank[k][ix].to(device) for k in ['joints_camera21','normalized_residual']};self.proposals['crop_uv']=geometry['crop_uv'][ix].to(device);self.labels={k:target[k].to(device) for k in ['error_mm','delta_error_mm','reference_good','harm_on_good']};self.train=[j for j,i in enumerate(self.indices) if self.rows[i]['role']=='train'];self.select=[j for j,i in enumerate(self.indices) if self.rows[i]['role']=='dev_select'];assert len(self.train)==1330 and len(self.select)==25
    def get(self,ids):
        b=self.batch([self.indices[i] for i in ids]);c={k:v[ids] for k,v in self.proposals.items()};target={k:v[ids] for k,v in self.labels.items()};return b,c,target
