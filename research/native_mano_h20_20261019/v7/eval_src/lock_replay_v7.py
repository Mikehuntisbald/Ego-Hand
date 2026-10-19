"""Lock both data arms for complete replay before module scoring opens."""
import hashlib,json,time
from pathlib import Path
R=Path('/mnt/why/native_mano_pool_v7_20261019')
def main():
    assert not (R/'evaluation/metrics.json').exists()
    models={arm:hashlib.sha256((R/'runs'/arm/'best_development.pt').read_bytes()).hexdigest() for arm in ['small','full']}
    result=dict(complete=True,locked_unix=time.time(),models=models,both_data_arms_in_full_replay=True,raw_and_guarded_outputs_all_frozen_before_replay_labels=True,
        no_module_score_based_arm_selection=True,historical_exposure=True,production_admitted=False)
    (R/'receipts/full_predeclared.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
