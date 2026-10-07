"""Freeze the verified raw adapter and auxiliary encoder assets."""
import hashlib,json,time
from pathlib import Path
from hand3d_v8_common import V7,save
import spatial_rgb_common as s
RUN=V7.parent/'adaptive_projection_v14/dit_dense';CODE=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert json.loads((RUN/'raw_inference_checks.json').read_text())['passed']
    assert json.loads((RUN/'fourth_results.json').read_text())['passed']
    names=['infer_hand3d_v14.py','encode_hand3d_dense_v14.py','wilor_eval_common.py','spatial_rgb_common.py','offline_rgb_encoder.py','cache_dit_v3.py']
    assets=[s.RUN/'sealed/rgb_probe.pt',V7.parent/'natural_reliability_v4/sealed/risk_projection.pt',s.common.RUN/'assets/wilor_final.mirror.ckpt',s.common.RUN/'assets/model_config.yaml',s.common.RUN/'assets/MANO_RIGHT.pkl',s.common.RUN/'assets/mano_mean_params.npz']
    receipt=dict(created_unix=time.time(),raw_checks_passed=True,model_and_policy_seal=str(RUN/'fourth_seal.json'),code_sha256={n:sha(CODE/n) for n in names},asset_sha256={str(p):sha(p) for p in assets})
    save(RUN/'deployment_seal.json',receipt);print(json.dumps(dict(sealed=True,code_files=len(names),assets=len(assets))),flush=True)

if __name__=='__main__':main()
