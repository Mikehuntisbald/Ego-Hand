"""Select a stricter score threshold on sealed development detections only."""
import json,collections
from pathlib import Path
import numpy as np
from prepare_paired_domain_v51 import PAIRED

def main():
    folder=PAIRED/'detector';rows={r['id']:r for r in map(json.loads,(PAIRED/'domain_records.jsonl').read_text().splitlines()) if r['split']=='dev'}
    sealed=json.loads((folder/'selected_dev_predictions.json').read_text());baseline=json.loads((folder/'baseline_dev.json').read_text());history=[]
    for threshold in [.05,.1,.15,.2,.25,.3,.4,.5]:
        statistics={}
        for prediction in sealed:
            row=rows[prediction['id']];gt=np.asarray([h['box_xyxy'] for h in row['hands']]);boxes=[b for b,s in zip(prediction['boxes'],prediction['scores']) if s>=threshold];matched=set()
            for box in boxes:
                if not len(gt):continue
                box=np.asarray(box);intersection=np.maximum(np.minimum(gt[:,2:],box[2:])-np.maximum(gt[:,:2],box[:2]),0).prod(-1);union=np.maximum(gt[:,2:]-gt[:,:2],0).prod(-1)+np.maximum(box[2:]-box[:2],0).prod()-intersection;ov=intersection/np.maximum(union,1e-9)
                for i in matched:ov[i]=0
                i=int(ov.argmax())
                if ov[i]>=.5:matched.add(i)
            s=statistics.setdefault(row['dataset'],dict(hands=0,matched=0,predicted=0,images=0));s['hands']+=len(gt);s['matched']+=len(matched);s['predicted']+=len(boxes);s['images']+=1
        for s in statistics.values():s.update(coverage=s['matched']/max(s['hands'],1),precision=s['matched']/max(s['predicted'],1))
        native=statistics['hot3d_preservation'];ego=statistics['egohands'];feasible=native['matched']>=baseline['hot3d_preservation']['matched'] and ego['coverage']>=baseline['egohands']['box_coverage_IoU50'] and ego['precision']>=baseline['egohands']['precision_IoU50']
        score=2*ego['precision']+ego['coverage']+statistics['cppe5']['coverage']+statistics['surgical_hands']['coverage']
        history.append(dict(threshold=threshold,statistics=statistics,feasible=feasible,score=score))
    selected=max([x for x in history if x['feasible']],key=lambda x:x['score'])
    (folder/'operating_point.json').write_text(json.dumps(dict(selected=selected,history=history,selection_split='Paired development only',preserve_native_matched_count=True,baseline_threshold=.05,external_test_used=False),indent=2));print(json.dumps(selected),flush=True)

if __name__=='__main__':main()
