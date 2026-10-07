import json,re,sys
from pathlib import Path
from verify_matched_dit_report_v43 import main as verify_geometry


def main():
    verify_geometry();root=Path(sys.argv[1]);page=(root/'report.html').read_text(encoding='utf-8')
    summary=json.loads((root/'full_summary.json').read_text());hard=json.loads((root/'hard_restore_effect.json').read_text())
    payload=json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',page,re.S).group(1))
    assert len(summary['methods'])==15
    assert '全部运动阈值×3' in page and '仅加速度阈值×2' in page and 'S.references' not in page
    assert hard['changed_frames_over_001mm']==hard['changed_frames_without_any_gt']+hard['changed_frames_with_any_gt']
    assert hard['max_all_points_change_mm']>200 and hard['changed_frames_without_any_gt']==30
    for key,value in summary['configs']['strict'].items():
        if key.endswith('speed'):assert summary['configs']['acc_x2'][key]==value
        if key.endswith('acc'):assert summary['configs']['acc_x2'][key]==2*value
    assert summary['reports']['strict']['coherence']['spurious_jump_pairs']==0
    assert summary['reports']['acc_x2']['coherence']['spurious_jump_pairs']==0
    assert summary['reports']['motion_x3']['coherence']['spurious_jump_pairs']==10
    assert {'jump_resolved','motion_recovered','accuracy_loss'}.issubset({c['id'] for c in payload['cases']})
    verification=json.loads((root/'verification.json').read_text());assert verification['passed']
    check=json.loads((root/'offline_checks.json').read_text());check.update(honest_constraint_labels=True,
        GT_unmatched_frame_changes_reported=True,strict_caps_and_acceleration_trial_correct=True,
        actual_jump_introduction_and_motion_recovery_included=True,browser_limit='Browser interaction not tested')
    (root/'offline_checks.json').write_text(json.dumps(check,indent=2));print(json.dumps(check))


if __name__=='__main__':main()
