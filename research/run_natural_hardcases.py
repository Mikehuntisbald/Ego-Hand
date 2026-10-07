import json,subprocess,sys,hashlib,tarfile,shutil
from pathlib import Path
import spatial_rgb_common as s
from natural_reliability import RUN
import numpy as np,torch
torch.set_num_threads(4);code=Path(__file__).resolve().parent;folder=RUN/'delivery'
items=json.loads((s.RUN/'natural_finger_audit/reviewed_examples.json').read_text());records,index=s.records_and_index();w=torch.load(s.OLD/'windows.pt',weights_only=False)
tracks=[];centers=[]
for item in items:
    i=item['window_index'];frames=[];center=None
    for t,fid in enumerate(index['feature_ids'][i].tolist()):
        if not fid:continue
        r=records[fid-1];available=w['observed'][i,t].tolist()
        if t==8:center=len(frames)
        frames.append(dict(image=r['image'],camera=r['camera'],clip=r['clip'],box_xyxy=r['box'],box_confidence=r['score'],timestamp_s=float(w['dt'][i,t]),
            xy_px=[(w['xy'][i,t,j]*1408).tolist() if available[j] else None for j in range(20)],available=available,confirmed=[False]*20))
    assert center is not None;tracks.append(dict(id=f"natural_case_{item['id']}",frames=frames));centers.append(center)
s.save(folder/'hardcase_input.json',dict(image_size=[1408,1408],tracks=tracks))
subprocess.run([sys.executable,str(code/'infer_natural_reliability.py'),'--input',str(folder/'hardcase_input.json'),'--output',str(folder/'hardcase_output.json')],cwd=code,check=True)
out=json.loads((folder/'hardcase_output.json').read_text());path=RUN/('predictions_final.npz' if (RUN/'predictions_final.npz').exists() else 'predictions.npz');expected=np.load(path);lookup={int(i):j for j,i in enumerate(expected['window_indices'])}
checks=[]
for item,center,tr in zip(items,centers,out['tracks']):
    i=item['window_index'];ix=lookup[i];pts=tr['frames'][center]['points'];xy=np.array([p['xy_px'] for p in pts])/1408
    delta=np.linalg.norm(xy-expected['prediction'][ix],axis=-1)*1408;assert delta.max()<1.,delta.max()
    m=w['valid'][i].numpy()&w['observed'][i,8].numpy();m[5]=False
    error=np.linalg.norm(xy-w['gt'][i].numpy(),axis=-1)*1408
    checks.append(dict(id=item['id'],frame_count=len(tr['frames']),center_frame_index=center,mean_finger_error_px=float(error[m].mean()),
        max_difference_vs_cached_px=float(delta.max()),corrected_points=sum(p['correction_applied'] for p in pts),
        specially_flagged_points=sum(p['needs_special_review'] for p in pts),all_automatic_points_require_review=all(p['review_required'] for p in pts)))
s.save(folder/'hardcase_cli_checks.json',checks)
shutil.copy2(code/'run_natural_hardcases.py',folder/'code/run_natural_hardcases.py')
shutil.copy2(code/'DEVELOPMENT_NOTES.txt',folder/'DEVELOPMENT_NOTES.txt')
s.save(folder/'manifest.json',{str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.rglob('*') if p.is_file() and p.name!='manifest.json'})
archive=RUN/'natural_reliability_v4.tar.gz'
with tarfile.open(archive,'w:gz') as tar:tar.add(folder,arcname='natural_reliability_v4')
s.save(RUN/'delivery_archive.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()))
print(json.dumps(checks,indent=2))
