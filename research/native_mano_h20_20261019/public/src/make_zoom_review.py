"""Readable natural finger crops from already scored, frozen predictions."""
import argparse,json,sys,tarfile,html
from pathlib import Path
import numpy as np,cv2,torch
BASE=Path('/mnt/why/public_headhand_h20_20261019')
sys.path.insert(0,str(BASE/'vendor/Ego-Hand/research'))
import wilor_eval_common as common
from geometry import camera,transform,load_frames
from hand_tracking_toolkit.dataset import decode_hand_pose,HandSide
from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json,forward_kinematics
EDGE=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-root',type=Path,required=True);a=ap.parse_args();root=a.run_root
    review=root/'outputs/review_clip2102';assert (review/'metrics.json').exists()
    rows=json.loads((root/'outputs/wilor_clip2102/observations.json').read_text())['frames'];frames=load_frames(root/'data/clip-002102')
    baseline=np.load(root/'outputs/mano_baseline_clip2102/parameters.npz');dit=np.load(root/'outputs/mano_dit_clip2102/parameters.npz')
    matches=json.loads((review/'associations.json').read_text());gt={};uv_gt={}
    with tarfile.open(root/'assets/clip-002102-full.tar') as tar:
        shape=from_json(json.load(tar.extractfile('__hand_shapes.json__'))['umetrack'])
        for match in matches:
            i=match['observation'];fi=match['frame'];frame=frames[fi]
            poses=decode_hand_pose(json.load(tar.extractfile(f'{frame["id"]}.hands.json')))
            pose=poses[HandSide.RIGHT if match['side']=='right' else HandSide.LEFT].umetrack
            with torch.no_grad():world=forward_kinematics(pose,shape,requires_mesh=False)[0].numpy()
            cam=transform(json.load(tar.extractfile(f'{frame["id"]}.cameras.json'))['214-1']['T_world_from_camera'])
            gt[i]=(world-cam[:3,3])@cam[:3,:3]
            uv_gt[i]=camera(frame['cameras']['214-1']['calibration']).eye_to_window(gt[i])
    indices=list(gt);ea={i:np.linalg.norm(baseline['joints_camera'][i,common.MAPPING]-gt[i],axis=-1).mean()*1000 for i in indices}
    eb={i:np.linalg.norm(dit['joints_camera'][i,common.MAPPING]-gt[i],axis=-1).mean()*1000 for i in indices}
    selections=[]
    for label,rank in [('DiT损伤较大',sorted(indices,key=lambda i:eb[i]-ea[i],reverse=True)),('DiT改善较大',sorted(indices,key=lambda i:ea[i]-eb[i],reverse=True)),('仍然困难',sorted(indices,key=lambda i:eb[i],reverse=True))]:
        chosen=[]
        for i in rank:
            if any(abs(rows[i]['frame_index']-rows[j]['frame_index'])<10 and rows[i]['track_id']==rows[j]['track_id'] for j in chosen):continue
            chosen.append(i);selections.append((label,i))
            if len(chosen)==4:break
    cards=[]
    for n,(label,i) in enumerate(selections):
        r=rows[i];fi=r['frame_index'];photo=cv2.imread(r['image']);H,W=photo.shape[:2];box=np.array(r['box_xyxy']);center=(box[:2]+box[2:])/2;size=max(box[2:]-box[:2])*1.35
        x0=max(0,int(center[0]-size/2));x1=min(W,int(center[0]+size/2));y0=max(0,int(center[1]-size/2));y1=min(H,int(center[1]+size/2));panels=[]
        native=camera(frames[fi]['cameras']['214-1']['calibration'])
        for name,points,color in [('RGB',None,None),('BA',baseline['joints_camera'][i],(255,130,0)),('DiT + BA',dit['joints_camera'][i],(180,0,255))]:
            image=photo.copy()
            if points is not None:
                uv=native.eye_to_window(points)
                for p,q in EDGE:cv2.line(image,tuple(np.rint(uv[p]).astype(int)),tuple(np.rint(uv[q]).astype(int)),color,2,cv2.LINE_AA)
                for p in uv_gt[i]:cv2.circle(image,tuple(np.rint(p).astype(int)),3,(0,255,0),-1)
            crop=image[y0:y1,x0:x1];scale=480/max(crop.shape[:2]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
            panel=np.full((540,500,3),28,np.uint8);panel[50:50+crop.shape[0],:crop.shape[1]]=crop
            cv2.putText(panel,name,(12,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2);panels.append(panel)
        name=f'zoom_{n:02}_frame{fi:06}_track{r["track_id"]}.jpg';cv2.imwrite(str(review/'images'/name),np.concatenate(panels,axis=1))
        caption=f'{label}；frame {fi} / track {r["track_id"]}；BA {ea[i]:.1f} → DiT+BA {eb[i]:.1f} mm'
        cards.append(f'<figure><img src="images/{name}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    section='<h2>自然手部放大图</h2><p>真实RGB裁图；绿色点为GT20，彩色线为MANO21预测。只按封存后的误差排序展示，未用GT挑选推理候选。</p>'+''.join(cards)
    path=review/'report.html';text=path.read_text();text=text.replace('<details>',section+'<details>',1);path.write_text(text)
    (review/'zoom_cases.json').write_text(json.dumps([dict(category=l,observation=i,BA_mm=ea[i],DiT_mm=eb[i]) for l,i in selections],ensure_ascii=False,indent=2))
    print(json.dumps(dict(complete=True,cases=len(cards))))

if __name__=='__main__':main()
