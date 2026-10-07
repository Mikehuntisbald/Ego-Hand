"""Scientific overlays with a shared crop; preserve full-frame originals."""
import json,collections,shutil
from pathlib import Path
import torch,numpy as np,cv2
from hand3d_v8_common import V7
from evaluate_joint_kinematic_v30 import EDGES
import spatial_rgb_common as s

def main():
    torch.set_num_threads(4);root=V7.parent;rows=json.loads((root/'acceleration_validation_v46/fresh_rows.json').read_text())
    windows=torch.load(root/'acceleration_validation_v46/fresh_data.pt',weights_only=False,mmap=True)['rows'];centers={r['row_index'] for r in windows}
    gt=torch.zeros(len(rows),20,3);valid=torch.zeros(len(rows),20,dtype=torch.bool)
    for i,r in enumerate(rows):
        if r['matched']:gt[i]=torch.tensor(r['gt']);valid[i]=torch.isfinite(gt[i]).all(-1)
    mask=valid.clone();mask[:,5]=False;eligible=torch.tensor([i for i,r in enumerate(rows) if i in centers and valid[i].all()])
    error=lambda p:(((p-p[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1)*1000*mask).sum(-1)/mask.sum(-1).clamp_min(1)
    for name in ['module_ablation_v49_20261007','decoder_ablation_v49_20261007']:
        run=root/name;cases=json.loads((run/'review/cases.json').read_text());full=torch.load(run/'full/result.pt',weights_only=False,map_location='cpu')['prediction'];fe=error(full)
        overview=run/'review/overview';overview.mkdir(exist_ok=True);cache={}
        for case in cases:
            variant=case['variant']
            if variant not in cache:cache[variant]=torch.load(run/variant/'result.pt',weights_only=False,map_location='cpu')['prediction']
            p=cache[variant];delta=error(p)-fe;choice=delta[eligible].argmax() if case['kind']=='removed_worse' else delta[eligible].argmin();i=int(eligible[choice]);r=rows[i]
            assert r['sequence']==case['sequence'] and r['frame']==case['frame'];image=cv2.imread(r['image']);cam=s.common.from_json(r['camera'])
            values=[('full',full,(50,220,70)),(variant,p,(30,150,255)),('GT',gt,(230,180,30))]
            uv=[cam.eye_to_window(q[i].numpy()) for _,q,_ in values];xy=np.concatenate(uv);xy=xy[np.isfinite(xy).all(1)]
            lo=xy.min(0);hi=xy.max(0);center=(lo+hi)/2;size=max(hi-lo)*1.8;size=max(size,180.)
            h,w=image.shape[:2];size=min(size,min(h,w));x=int(np.clip(center[0]-size/2,0,w-size));y=int(np.clip(center[1]-size/2,0,h-size));side=int(size)
            panels=[]
            for (label,q,color),points in zip(values,uv):
                crop=image[y:y+side,x:x+side].copy();points=points-np.asarray([x,y])
                for u,v in EDGES:
                    if np.isfinite(points[[u,v]]).all():cv2.line(crop,tuple(np.clip(points[u],-4000,4000).astype(int)),tuple(np.clip(points[v],-4000,4000).astype(int)),color,max(1,side//140))
                for pt in points:
                    if np.isfinite(pt).all():cv2.circle(crop,tuple(np.clip(pt,-4000,4000).astype(int)),max(2,side//100),color,-1)
                crop=cv2.resize(crop,(480,480));cv2.rectangle(crop,(0,0),(480,35),(20,20,20),-1);cv2.putText(crop,label,(10,25),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2);panels.append(crop)
            dest=run/'review'/case['image'];old=overview/case['image']
            if not old.exists():shutil.copy2(dest,old)
            cv2.imwrite(str(dest),np.concatenate(panels,1));case['crop_xywh']=[x,y,side,side];case['original_overview']='overview/'+case['image']
        (run/'review/cases.json').write_text(json.dumps(cases,indent=2))
        print(json.dumps(dict(run=name,cases=len(cases),shared_crops=True,full_frames_preserved=True)),flush=True)
if __name__=='__main__':main()
