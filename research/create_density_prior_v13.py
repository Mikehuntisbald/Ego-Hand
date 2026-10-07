"""Snapshot the audited prior, changing only the velocity's timestamp floor."""
import inspect,hashlib
from pathlib import Path
from hand3d_trajectory_v9 import TrajectoryHand3D
from hand3d_v8_common import V7,save
code=Path(__file__).resolve().parent;source=inspect.getsource(TrajectoryHand3D.loss);source=__import__('textwrap').dedent(source)
old="dt=(b['dt'][:,1:]-b['dt'][:,:-1]).clamp_min(.05)"
new="dt=(b['dt'][:,1:]-b['dt'][:,:-1]).clamp_min(.001)"
assert source.count(old)==1;source=source.replace(old,new)
header='import torch\nfrom torch.nn import functional as F\nfrom hand3d_temporal_v7 import pack,unpack,WRIST,EDGES\n'
dest=code/'density_prior_loss_v13.py';dest.write_text(header+source)
save(V7.parent/'aligned_density_v13/prior_loss_snapshot.json',dict(original_sha256=hashlib.sha256(inspect.getsource(TrajectoryHand3D.loss).encode()).hexdigest(),snapshot_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),change='Only50ms floor replaced by1ms numerical guard; original actual timestamps and repaired target validity remain'))
print(str(dest),flush=True)
