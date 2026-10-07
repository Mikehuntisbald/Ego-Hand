"""Validate local scientific report without browser policy workarounds."""
import argparse,json,math,re
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('directory');a=ap.parse_args();root=Path(a.directory)
    page=(root/'report.html').read_text(encoding='utf-8');payload=json.loads(re.search(r'<script id="report-data" type="application/json">(.*?)</script>',page,re.S).group(1))
    count=0;maxdelta=0
    for c in payload['cases']:
        assert len(c['frames'])>=2 and 0<=c['focus']<len(c['frames'])
        for key in ['full_image','snapshot','curves']:assert(root/c[key]).is_file()
        for f in c['frames']:
            count+=1;assert(root/f['image']).is_file()
            for k in ['gt','v16','v39','v42']:
                assert len(f['points'][k])==len(f['pose'][k])==20
                for p in f['pose'][k]:assert len(p)==3 and all(x is None or math.isfinite(x) for x in p)
                if f['complete_gt']:
                    error=sum(math.dist(f['pose'][k][j],f['pose']['gt'][j]) for j in range(20) if j!=5)/19
                    delta=abs(error-f['relative_mm'][k]);maxdelta=max(maxdelta,delta);assert delta<.06
    co=payload['summary']['center_reports']['v42']['coherence'];assert co['spurious_jump_pairs']==co['bone_extreme_frames']==co['bone_flicker_gt_under1_pred_over10']==0
    executable=re.findall(r'<script>(.*?)</script>',page,re.S);assert len(executable)==1
    (root/'viewer_script.mjs').write_text(executable[0],encoding='utf-8')
    check=dict(passed=True,cases=len(payload['cases']),frames=count,all_assets_present=True,
        max_displayed_error_delta_mm=maxdelta,finite_geometry=True,stability_summary_matches=True,
        browser_interaction_checked=False,browser_limit='file:// navigation restricted; no alternate browser/hosting workaround used')
    (root/'offline_checks.json').write_text(json.dumps(check,indent=2));print(json.dumps(check))


if __name__=='__main__':main()
