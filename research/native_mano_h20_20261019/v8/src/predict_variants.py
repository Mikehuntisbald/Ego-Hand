"""Freeze all depth candidates without importing or reading hand GT."""
import hashlib,json
from pathlib import Path
import cv2,numpy as np
from depth_common import VIEW,build_views,select,anchor
from geometry import load_frames
R=Path('/mnt/why/native_depth_audit_v8_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert not (R/'predictions_frozen.json').exists();seal=json.loads((R/'stereo_frozen.json').read_text());protocol=json.loads((R/'protocol.json').read_text())
    for name,expected in seal['files_sha256'].items():assert sha(R/'stereo_grid'/name)==expected
    frames=load_frames(VIEW/'data/clip-002102');rows=json.loads((VIEW/'outputs/wilor_clip2102/observations.json').read_text())['frames'];surface=np.load(VIEW/'outputs/wilor_clip2102/surface.npz');rf=json.loads((VIEW/'outputs/rf_clip2102/predictions.json').read_text())['frames']
    original=json.loads((VIEW/'outputs/depth_clip2102/depth.json').read_text())['observations'];out=[]
    for fi in protocol['frames']:
        frame=frames[fi];frame['frame_index']=fi;ids=[i for i,r in enumerate(rows) if r['frame_index']==fi];views=build_views(frame,ids,rows,R/'stereo_grid')
        with np.load(rf[fi]['output']) as v:masks=v['masks'];own=masks&~(masks.sum(0)>1)[None]
        for i in ids:
            vertices=surface['vertices'][i];wrist=surface['joints'][i,0];assert wrist[2]>.08
            candidates={}
            for erosion in [0,4]:
                mask=own[rows[i]['detection']].astype(np.uint8)
                if erosion:mask=cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
                pts=[select(v,mask.astype(bool))[0] for v in views];points=np.concatenate(pts)
                baseline,_,_=anchor(points,vertices,surface['faces'],wrist[2]);key='erode4' if erosion else 'baseline';candidates[key]=baseline
                current=float(wrist[2]);history=[];candidate=None
                for step in range(5):
                    shifted=vertices+wrist[None]*(current/float(wrist[2])-1.)
                    candidate,_,_=anchor(points,shifted,surface['faces'],current);history.append(dict(step=step,current_m=current,**candidate))
                    if not candidate['valid']:break
                    delta=abs(candidate['depth_m']-current);current=candidate['depth_m']
                    if delta<1e-4:break
                iteration=dict(candidate);iteration['iteration_history']=history;iteration['fallback_to_baseline']=not iteration['valid']
                if not iteration['valid']:iteration.update(baseline)
                candidates['erode4_iter5' if erosion else 'iter5']=iteration
            old=next(x for x in original if x['observation']==i);new=candidates['baseline'];assert new['valid']==old['valid']
            assert new['samples']==old['samples']
            if new['valid']:assert abs(new['depth_m']-old['wrist_depth_m'])<1e-6,(fi,i,new,old)
            out.append(dict(observation=i,frame_index=fi,track_id=rows[i]['track_id'],predicted_right=rows[i].get('right',rows[i].get('is_right')),original=old,candidates=candidates))
        print(json.dumps(dict(frame=fi,observations=len(out))),flush=True)
    target=R/'predictions.json';target.write_text(json.dumps(dict(complete=True,GT_used=False,observations=out),indent=2))
    sources={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    (R/'predictions_frozen.json').write_text(json.dumps(dict(complete=True,GT_used=False,predictions_sha256=sha(target),stereo_frozen_sha256=sha(R/'stereo_frozen.json'),source_sha256=sources,baseline_reproduced=True,production_admitted=False),indent=2))
if __name__=='__main__':main()
