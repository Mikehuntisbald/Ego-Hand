"""Check report geometry, honest method labels and short-context protocol."""
import json,math,re,sys
from pathlib import Path
from verify_matched_dit_report_v43 import main as verify_geometry


def main():
    verify_geometry()
    root=Path(sys.argv[1]);page=(root/'report.html').read_text(encoding='utf-8')
    payload=json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',page,re.S).group(1))
    summary=json.loads((root/'full_summary.json').read_text());extra=0.;checked=0
    for case in payload['cases']:
        j=case['focus_joint']
        for frame in case['frames']:
            for method in ['gt','v16','v39','v42']:
                expected=frame['focus_joint_relative_mm'][method]
                if expected is None:continue
                p,g=frame['pose'][method][j],frame['pose']['gt'][j]
                if any(x is None for x in p+g):continue
                delta=abs(math.dist(p,g)-expected);extra=max(extra,delta);checked+=1
                assert delta<.06
    assert {'accuracy_loss','still_wrong'}.issubset({x['id'] for x in payload['cases']})
    assert '±4秒 DiT整段选择' in page and '±1.6秒 DiT整段选择' in page
    assert '匹配时序回归' not in page and 'S.references' not in page
    assert summary['context_coverage']['short_context']['max_observed_context_s']<=1.600003
    assert summary['context_coverage']['long_candidate_parameter_max_difference']==0
    assert len(summary['methods'])==6
    check=json.loads((root/'offline_checks.json').read_text())
    check.update(focus_point_error_checks=checked,max_focus_point_error_delta_mm=extra,
                 honest_window_labels=True,short_context_bounded=True,
                 accurate_point_failure_included=True,browser_limit='Browser interaction not tested')
    (root/'offline_checks.json').write_text(json.dumps(check,indent=2));print(json.dumps(check))


if __name__=='__main__':main()
