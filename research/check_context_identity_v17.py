"""Attach GT identities after frozen context choices; never choose with GT."""
import json
from pathlib import Path
import numpy as np
from hand3d_v8_common import V7,save
import spatial_rgb_common as s
RUN=V7.parent/'context_merge_v17'

def main():
    choices=json.loads((RUN/'audit_results.json').read_text());side_audit=json.loads((V7.parent/'side_consensus_v16/audit_results.json').read_text());summary={};annotation_cache={}
    for label,folder in [('train_development','dense_sampling_v13'),('retained_failures','hard_dense_v14')]:
        records=json.loads((V7.parent/folder/'fresh_rows.json').read_text());centres={r['window_index']:r['GT_side_diagnostic'] for r in side_audit['sets'][label]['rows']};known=wrong=unknown=0;rows=[]
        for case in choices['sets'][label]['changed_windows']:
            for addition in case['added']:
                r=records[addition['fid']-1];actual=None
                if r['matched']:
                    image=Path(r['image']);sp,seq,cid=image.relative_to(s.common.ROOT/'export/images').parts[:3];path=s.common.ROOT/'export/annotations'/sp/seq/(cid+'.jsonl')
                    if path not in annotation_cache:annotation_cache[path]=[json.loads(line) for line in path.read_text().splitlines()]
                    frame=annotation_cache[path][r['frame']];differences=[float(np.max(np.abs(np.asarray(h['xyz_camera_m'])-np.asarray(r['gt'])))) for h in frame['hands']];j=int(np.argmin(differences));assert differences[j]<2e-6;actual=frame['hands'][j]['side']
                expected=centres[case['window_index']];mismatch=actual is not None and expected is not None and actual!=expected
                if actual is None or expected is None:unknown+=1
                else:known+=1;wrong+=mismatch
                rows.append(dict(window_index=case['window_index'],slot=addition['slot'],frame=addition['frame'],fid=addition['fid'],GT_center_side_diagnostic=expected,GT_context_side_diagnostic=actual,mismatch=mismatch,role=case['role']))
        summary[label]=dict(known_added_contexts=known,wrong_hand=wrong,unmatched_or_unknown=unknown,rows=rows)
    save(RUN/'identity_results.json',dict(complete=True,sets=summary,scope='GT attached only after choices to diagnose identities, not independent GTquality validation. Unknown contexts retained. No pose model/risk trained or recovery benefit claimed. Trainingtargets must use center-hand identity at every timestamp, not blindly follow the chosen observation GTmatch.'))
    print(json.dumps({k:{n:v for n,v in x.items() if n!='rows'} for k,x in summary.items()},indent=2),flush=True)

if __name__=='__main__':main()
