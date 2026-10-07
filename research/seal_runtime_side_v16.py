import hashlib,json,time
from pathlib import Path
from hand3d_v8_common import V7,save
import spatial_rgb_common as s
RUN=V7.parent/'side_native_v16';CODE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((RUN/'raw_inference_checks.json').read_text())['passed'];assert json.loads((RUN/'fifth_results.json').read_text())['passed']
    names=['complete_hand_tracks_v16.py','infer_hand3d_v16.py','infer_hand3d_v14.py','encode_hand3d_dense_v14.py','audit_side_consensus_v16.py','dit_v3_inference.py','wilor_eval_common.py','spatial_rgb_common.py','offline_rgb_encoder.py','cache_dit_v3.py','prepare_pose_training.py','compare_detectors.py']
    assets=[s.RUN/'sealed/rgb_probe.pt',V7.parent/'natural_reliability_v4/sealed/risk_projection.pt',V7.parent/'detector_compare_wilor_20261003/wilor_detector.pt',s.common.RUN/'assets/wilor_final.mirror.ckpt',s.common.RUN/'assets/model_config.yaml',s.common.RUN/'assets/MANO_RIGHT.pkl',s.common.RUN/'assets/mano_mean_params.npz',s.common.ROOT/'weights/yolo26s.pt',V7.parent/'dit_lowconfidence_v1/coarse_fine/best.pt']
    save(RUN/'deployment_seal.json',dict(created_unix=time.time(),raw_checks_passed=True,model_policy_seal=str(RUN/'fifth_seal.json'),code_sha256={n:sha(CODE/n) for n in names},asset_sha256={str(p):sha(p) for p in assets}))
    print(json.dumps(dict(sealed=True,code_files=len(names),assets=len(assets))),flush=True)
if __name__=='__main__':main()
