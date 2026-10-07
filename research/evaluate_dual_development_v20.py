"""Reuse the identical operating comparison for the matched dual-RGB arms."""
import hashlib
from pathlib import Path
from hand3d_v8_common import V7, save

def main():
    code = Path(__file__).resolve().parent
    run = V7.parent/'dual_visual_v20'
    source = (code/'evaluate_canonical_development_v19.py').read_text()
    changes = {
        "RUN = V7.parent / 'canonical_native_v19'": "RUN = V7.parent / 'dual_visual_v20'",
        "'canonical': V7.parent / 'canonical_rgb_v19'": "'canonical': V7.parent / 'context_data_v17/control'",
        "DATA['canonical'] / 'prediction_only_encoding_plan.json'": "V7.parent / 'canonical_rgb_v19/prediction_only_encoding_plan.json'",
        "prior_control = torch.load(V7.parent/'context_native_v17/control/uniform_adaptive/calibration.pt',weights_only=False)": "prior_control = prior",
        "approved=reports['canonical']['approved_for_independent_test']": "approved=any(r['approved_for_independent_test'] for r in reports.values())",
        "selected_variant='canonical' if reports['canonical']['approved_for_independent_test'] else None": "selected_variant=next(iter(sorted([k for k,r in reports.items() if r['approved_for_independent_test']],key=lambda k:(-reports[k]['hard']['relative_bad_recovered20'],reports[k]['hard']['relative_mm'],reports[k]['final']['relative_mm']))),None)",
        "repeated_v17_control_max_raw_delta_mm=replay_delta": "dual_control_vs_v16_max_raw_delta_mm=replay_delta",
        "Same900updates/seed/initializer/3Dlabels/XYZ/slots/policy; RGBorientation/physicalpositions/risk differ.": "Same900updates/seed/initializer/3Dlabels/XYZ/slots/policy/risk/originalRGB/localization/addedcapacity; extra semantic memory RGBorientation/physicalpositions differ. Botharms eligible for independent validation only after hardrecovery improves.",
    }
    generated = source
    for before,after in changes.items():
        assert generated.count(before)==1,before
        generated = generated.replace(before,after)
    snapshot = run/'evaluator_snapshot.py'
    snapshot.write_text(generated)
    save(run/'evaluator_provenance.json',dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_sha256=hashlib.sha256(generated.encode()).hexdigest(),replacements=changes))
    ns = dict(__name__='dual_visual_evaluation_v20',__file__=str(snapshot))
    exec(compile(generated,str(snapshot),'exec'),ns)
    ns['main']()

if __name__=='__main__':
    main()
