"""GT-free final BA displacement gates against the matched public BA control."""
import json
from pathlib import Path
import numpy as np
R=Path('/mnt/why/native_mano_dit_v2_20261019');VIEW=R/'coherent_frontend/outputs'

def main():
    base=np.load(VIEW/'mano_baseline_clip2102/parameters.npz');reference=json.loads((VIEW/'mano_baseline_clip2102/optimization.json').read_text())
    results={}
    for arm in ['regression','linear_protected','cosine_dit','cosine_protected']:
        proposed=np.load(VIEW/f'ba_{arm}/parameters.npz');report=json.loads((VIEW/f'ba_{arm}/optimization.json').read_text())
        for key in ['timestamp_ns','frame_index','track_id','right','group']:assert np.array_equal(base[key],proposed[key]),key
        displacement=np.linalg.norm(proposed['joints_camera']-base['joints_camera'],axis=-1)*1000
        overall=dict(RMSE_mm=float(np.sqrt(np.mean(displacement**2))),P95_mm=float(np.quantile(displacement,.95)),max_mm=float(displacement.max()),mean_mm=float(displacement.mean()))
        reproj=report['reprojection_mean_px']-reference['reprojection_mean_px']
        passes=overall['RMSE_mm']<=5 and overall['P95_mm']<=10 and overall['max_mm']<=30 and reproj<=.2
        results[arm]=dict(displacement=overall,reprojection_increase_px=reproj,timestamp_identity_chirality_unchanged=True,public_development_displacement_gate_pass=passes,
            production_admission=False,reason='same public development control only; not the private production reference or multi-sequence generalization')
    (R/'receipts/final_BA_displacement_audit.json').write_text(json.dumps(dict(complete=True,GT_used=False,arms=results),indent=2));print(json.dumps(results))

if __name__=='__main__':main()
