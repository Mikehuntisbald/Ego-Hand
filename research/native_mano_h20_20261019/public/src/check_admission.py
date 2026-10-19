"""Report explicit DiT-vs-BA regression gates; never modifies inference outputs."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',required=True,type=Path);a=ap.parse_args();root=a.run_root
    base=root/'outputs/mano_baseline_clip2102';dit=root/'outputs/mano_dit_clip2102';review=root/'outputs/review_clip2102'
    x=np.load(base/'parameters.npz');y=np.load(dit/'parameters.npz')
    same={key:bool(np.array_equal(x[key],y[key])) for key in ['frame_index','track_id','timestamp_ns','right']}
    displacement=np.linalg.norm(x['joints_world']-y['joints_world'],axis=-1)*1000
    gates=dict(RMSE_mm=float(np.sqrt(np.mean(displacement**2))),P95_mm=float(np.quantile(displacement,.95)),max_mm=float(displacement.max()))
    b=json.loads((base/'optimization.json').read_text());d=json.loads((dit/'optimization.json').read_text());metrics=json.loads((review/'metrics.json').read_text())
    reprojection_increase=d['reprojection_mean_px']-b['reprojection_mean_px']
    passed=gates['RMSE_mm']<=5 and gates['P95_mm']<=10 and gates['max_mm']<=30 and reprojection_increase<=.2 and all(same.values())
    result=dict(complete=True,safe_default_admitted=False,version_displacement=gates,version_displacement_limits=dict(RMSE_mm=5,P95_mm=10,max_mm=30),
        reprojection_increase_px=reprojection_increase,reprojection_increase_limit_px=.2,identity_timeline_chirality_same=same,
        regression_thresholds_passed=bool(passed),correct_point_harm=metrics['DiT_vs_BA_protection_camera'],
        heldout_sequence_validation=False,sequence_CI_available=False,
        scope='These are version-difference gates against this public BA baseline, not absolute GT accuracy targets or private production admission.',
        reasons=['Only one reused development sequence; training overlap unestablished','Native accuracy and recovery do not ensure original correct points are protected']+([] if passed else ['Version displacement or reprojection regression limits failed']))
    (review/'admission.json').write_text(json.dumps(result,indent=2))
    report=review/'report.html';text=report.read_text();section='<h2>准入状态：未通过</h2><p>DiT保持实验分支。均值改善不能替代原正确点保护、版本位移保护线和独立留出验证。</p>'
    text=text.replace('<h1>H20公开链路与DiT对照</h1>','<h1>H20公开链路与DiT对照</h1>'+section,1);report.write_text(text)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
