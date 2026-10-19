"""Declare full-chain compatible arm before this iteration's scoring."""
import hashlib,json,time
from pathlib import Path
R=Path('/mnt/why/native_mano_shape_v6_20261019')
def main():
    assert not (R/'evaluation/metrics.json').exists()
    p=R/'runs/compatible/best_development.pt'
    result=dict(complete=True,locked_unix=time.time(),checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),arm='compatible',
        reason='Predeclared ordinary shape-compatible arm; test whether corrected pose transfers when BA optimizes shared shape',
        same_frontend_and_depth=True,same_BA=True,same_5_10_30_and_point2_gates=True,historical_diagnostic_replay=True,not_fresh_blind_test=True,production_admitted=False)
    (R/'receipts/full_predeclared.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
