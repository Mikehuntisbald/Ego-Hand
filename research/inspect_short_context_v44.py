import collections,json
import torch
from hand3d_v8_common import V7
from prepare_joint_mano_v28 import records_bank
from temporal_window_v44 import ObservationSampler,SHORT_OFFSETS,window_statistics

root=V7.parent
d=torch.load(root/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
primary,full=records_bank(d)
print(json.dumps({'data':{k:list(v.shape) for k,v in d.items() if torch.is_tensor(v)},'records':len(full),'primary':len(primary),'roles':dict(collections.Counter(d['roles']))}),flush=True)
sampler=ObservationSampler(primary,full)
f,dt=sampler.sample(d['feature_ids'][:,8].tolist())
assert torch.equal(f[:,8],d['feature_ids'][:,8])
print(json.dumps({'short':window_statistics(f,dt),'offsets_frames':SHORT_OFFSETS}),flush=True)
targets=torch.load(root/'parameter_kinematic_v31/refined_targets.pt',weights_only=False,mmap=True)
print(json.dumps({'targets':{k:list(v.shape) for k,v in targets.items() if torch.is_tensor(v)}}),flush=True)
