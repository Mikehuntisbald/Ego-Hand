"""CPU readiness and explicit no-instance-mask legacy baselines."""
import json,shutil
from pathlib import Path
import torch
from prepare_fair_v54 import ROOT,OLD,E,INITIAL,save,sha
def main():
    expected=[INITIAL,E/'matched_parameter_v43/dit/best.pt',E/'online_rgb_iterative_v48/protected/best.pt',E/'dit_lowconfidence_v1/detector/weights/best.pt',OLD/'paired_protocol/localizer_isolated_r2/best.pt']
    for arm in ['yolo_condition_control','rf_condition_adapt']:
        expected += [ROOT/arm/name for name in ['domain_masks.pt','domain_pixels.npy','native/targets.pt','native/risk.pt','native/pixels.npy','surgical_pose/targets.pt']]
    assert all(p.exists() for p in expected),[str(p) for p in expected if not p.exists()]
    assert not (ROOT/'controller_status.json').exists(),'Do not amend an active experiment'
    protocol=json.loads((ROOT/'protocol.json').read_text())
    archive=ROOT/'protocol_before_baseline_clarification.json'
    if not archive.exists():shutil.copy2(ROOT/'protocol.json',archive)
    protocol.update(formal_original_baselines=['Historical YOLO26 + WiLoR + v43 plain SemanticParameterHand; NO instance mask input or SAM2','Historical YOLO26 + WiLoR + v48 protected plain SemanticParameterHand; NO instance mask input or SAM2'],
                    detector_only_control='RF-DETR hand boxes + the SAME no-instance-mask v48 backend; segmentation predictions are ignored by this backend.',
                    auxiliary_training_control='yolo_v54_control is a separately labelled v51 mask-conditioned, same-budget training control. It is NOT the original v43-v48 baseline.',
                    box_association_not_instance_input='Rasterized predicted rectangles compute box IoU in a common tracker only. Legacy reconstruction receives no segmentation/mask features.')
    save(ROOT/'protocol.json',protocol)
    prepared=json.loads((ROOT/'prepared.json').read_text());prepared['protocol_sha256']=sha(ROOT/'protocol.json');prepared['baseline_clarification_before_training']=True;save(ROOT/'prepared.json',prepared)
    # The extra script is recorded before controller dispatch, while no GPU
    # worker/controller has started. Previously frozen files remain identical.
    src=Path(__file__);dst=ROOT/'code'/src.name
    if src.resolve()!=dst.resolve():shutil.copy2(src,dst)
    hashes=json.loads((ROOT/'code_hashes.json').read_text());hashes[src.name]=sha(dst);save(ROOT/'code_hashes.json',hashes)
    ck=torch.load(INITIAL,weights_only=False,map_location='cpu')
    assert ck['kind']=='dit' and 'visual_tail' in ck
    checks=dict(passed=True,assets=len(expected),shared_initial_step=ck['step'],shared_initial_sha256=sha(INITIAL),formal_legacy_masks=False,CPU_only=True,full_GPU_preflight_pending=True)
    save(ROOT/'CPU_readiness.json',checks);print(json.dumps(checks),flush=True)
if __name__=='__main__':main()
