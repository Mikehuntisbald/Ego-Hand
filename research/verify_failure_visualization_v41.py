"""Offline artifact validation; no browser launch or model changes."""
import argparse,json,math,re
from pathlib import Path


def distance(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');a=ap.parse_args();root=Path(a.directory)
    page=(root/'report.html').read_text(encoding='utf-8')
    payload=json.loads(re.search(r'<script type="application/json" id="report-data">(.*?)</script>',page,re.S).group(1))
    frames=0;max_error=0
    for case in payload['cases']:
        for key in ['full_image','snapshot','curves']:assert (root/case[key]).is_file(),case[key]
        for frame in case['frames']:
            frames+=1;assert (root/frame['image']).is_file(),frame['image']
            for key in ['gt','old','new','input','rejected']:
                assert len(frame['points'][key])==20 and len(frame['world_relative'][key])==20
            if not frame['accepted']:
                assert frame['points']['old']==frame['points']['new']
                assert frame['world_relative']['old']==frame['world_relative']['new']
            if frame['complete_gt']:
                gt=frame['world_relative']['gt']
                for key in ['old','new']:
                    error=sum(distance(gt[j],frame['world_relative'][key][j]) for j in range(20) if j!=5)/19
                    delta=abs(error-frame['metrics'][key]['relative_mm']);max_error=max(max_error,delta)
                    assert delta<.05,(case['id'],frame['frame'],key,delta)
        if case['id']=='lost_recovery':
            focus=case['frames'][case['focus']]
            assert focus['metrics']['old']['focus_relative_mm']<=10
            assert focus['metrics']['new']['focus_relative_mm']>20
        assert len(case['frames'])>=2
    executable=re.findall(r'<script>(.*?)</script>',page,re.S);assert len(executable)==1
    (root/'viewer_script.mjs').write_text(executable[0],encoding='utf-8')
    check=dict(passed=True,cases=len(payload['cases']),frames=frames,all_assets_present=True,
               displayed_world_geometry_matches_camera_relative_errors_max_mm=max_error,
               rejected_frames_match_v16_exactly=True,selected_recovery_loss_threshold_verified=True,
               browser_interaction_checked=False,
               browser_limit='Agent browser policy rejects file:// navigation; no alternate browser or hosting workaround attempted.')
    (root/'offline_checks.json').write_text(json.dumps(check,indent=2),encoding='utf-8')
    print(json.dumps(check,indent=2))


if __name__=='__main__':main()
