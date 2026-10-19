"""Post-seal native amodal image localization; no claim of 3D/visibility accuracy."""
import collections,hashlib,html,json,sys,tarfile,time
from pathlib import Path
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
R=Path('/mnt/why/native_rgb_observation_v9_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019');V3=Path('/mnt/why/native_mano_multiseq_v3_20261019')
sys.path.insert(0,str(R/'eval_src'));sys.path.insert(0,str(V3/'src'));sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/src')
from infer_observations import crop_uv
from geometry import camera
from evaluate_validation import iou,stats
ARMS=['WiLoR','spatial','spatial_blend','pooled','pooled_blend']
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def paired(delta,records):
    groups=sorted({r['sequence'] for r in records});sums=np.array([delta[[r['sequence']==g for r in records]].sum() for g in groups]);counts=np.array([delta[[r['sequence']==g for r in records]].size for g in groups]);rng=np.random.default_rng(190019);draw=rng.integers(len(groups),size=(10000,len(groups)));values=sums[draw].sum(1)/counts[draw].sum(1)
    return dict(clusters=len(groups),mean_delta_px=float(delta.mean()),CI95_px=np.quantile(values,[.025,.975]).tolist(),scope='historical six development recordings, three people; image pixels, not3D')
def main():
    cv2.setNumThreads(1);assert json.loads((R/'full_status.json').read_text())['complete'];frozen=json.loads((R/'observations_frozen.json').read_text())
    for name,h in frozen['seals'].items():
        p=R/name;assert sha(p)==h
        for n,v in json.loads(p.read_text())['files'].items():assert sha(p.parent/n)==v
    out=R/'evaluation';out.mkdir(exist_ok=False);(out/'images').mkdir();(out/'label_open_order.json').write_text(json.dumps(dict(unix=time.time(),observations_seal_sha256=sha(R/'observations_frozen.json'),full_BA_seal_sha256=sha(R/'full_pipeline/receipts/predictions_frozen.json'),historical_exposure=True),indent=2))
    GT=np.load(V7/'evaluation/sealed_scoring_arrays.npz')['GT_native_camera21'];records=[];truth=[];pred={a:[] for a in ARMS};canonical_gt=[];canonical_pred={a:[] for a in ['WiLoR','spatial','pooled']};risk={m:collections.defaultdict(list) for m in ['spatial','pooled']};counter=0
    for clip in json.loads((V3/'protocol.json').read_text())['clips']:
        view=Path(clip['view']);rows=json.loads((view/'outputs/wilor_clip2102/observations.json').read_text())['frames'];coarse=np.load(view/'outputs/wilor_clip2102/observations.npz');heads={m:np.load(R/f"predictions/clip-{clip['clip']:06}/{m}.npz") for m in ['spatial','pooled']}
        with tarfile.open(clip['archive']) as tar:
            for fi in range(150):
                hands=json.load(tar.extractfile(f'{fi:06}.hands.json'));cal=json.load(tar.extractfile(f'{fi:06}.cameras.json'))['214-1']['calibration'];targets=[]
                for side,h in hands.items():
                    if not h.get('umetrack_pose') or '214-1' not in h.get('boxes_amodal',{}):continue
                    box=np.array(h['boxes_amodal']['214-1']);box[[0,2]]=np.clip(box[[0,2]],0,1408);box[[1,3]]=np.clip(box[[1,3]],0,1408)
                    if np.maximum(box[2:]-box[:2],0).prod()>=64:targets.append((side,box))
                ids=[i for i,r in enumerate(rows) if r['frame_index']==fi]
                if not ids or not targets:continue
                overlap=np.array([[iou(np.array(rows[i]['box_xyxy']),t[1]) for t in targets] for i in ids]);a,b=linear_sum_assignment(1-overlap)
                for x,y in zip(a,b):
                    if overlap[x,y]<.1 or targets[y][0]!='right' or not hands['right'].get('mano_pose'):continue
                    i=ids[x];gt=GT[counter];counter+=1;uv=camera(cal).eye_to_window(gt);truth.append(uv);rot=coarse['crop_rotations'][i:i+1];f=coarse['crop_focals'][i:i+1];right=coarse['right'][i:i+1].astype(bool);cgt=crop_uv(gt[None],rot,f,right)[0];canonical_gt.append(cgt);canonical_pred['WiLoR'].append(crop_uv(coarse['xyz_camera'][i:i+1],rot,f,right)[0]);pred['WiLoR'].append(rows[i]['xy']);records.append(dict(clip=clip['clip'],sequence=clip['sequence'],subject=clip['subject'],frame=fi,observation=i,image=rows[i]['image'],box=rows[i]['box_xyxy']))
                    for mode,data in heads.items():
                        pred[mode].append(data['uv_head_native'][i]);pred[mode+'_blend'].append(data['uv_blended_native'][i]);canonical_pred[mode].append(data['crop_uv'][i]);risk[mode]['coarse_probability'].append(data['coarse_bad_probability'][i]);risk[mode]['new_probability'].append(data['new_bad_probability'][i]);risk[mode]['coarse_actual'].append((np.linalg.norm((data['coarse_fk_crop_norm'][i]-cgt)*[192,256],axis=-1)>8));risk[mode]['new_actual'].append((np.linalg.norm((data['crop_uv'][i]-cgt)*[192,256],axis=-1)>8));risk[mode]['weight'].append(data['blend_weight'][i])
    assert counter==894==len(GT);truth=np.asarray(truth);pred={k:np.asarray(v) for k,v in pred.items()};gtcrop=np.asarray(canonical_gt);canonical_pred={k:np.asarray(v) for k,v in canonical_pred.items()};inband=(gtcrop>=0).all(-1)&(gtcrop<=1).all(-1);native_inimage=(truth>=0).all(-1)&(truth<1408).all(-1);error={a:np.linalg.norm(v-truth,axis=-1) for a,v in pred.items()};base=error['WiLoR'];good=base<=5;hard=base>10
    metrics={a:dict(native_image_all21_px=stats(e.ravel()),native_image_framed_px=stats(e[native_inimage]),base_good5=int(good.sum()),good_harmed_over1px=int((good&(e>base+1)).sum()),good_to_over10=int((good&(e>10)).sum()),base_hard10=int(hard.sum()),recovered_to10=int((hard&(e<=10)).sum())) for a,e in error.items()}
    for a,p in canonical_pred.items():e=np.linalg.norm((p-gtcrop)*[192,256],axis=-1);metrics[a]['canonical_inband_crop_px']=stats(e[inband])
    calibration={}
    for m,d in risk.items():
        calibration[m]={}
        for prefix in ['coarse','new']:
            prob=np.asarray(d[prefix+'_probability']);actual=np.asarray(d[prefix+'_actual']);calibration[m][prefix]=dict(Brier=float(((prob-actual)**2).mean()),mean_probability=float(prob.mean()),actual_error_over8_fraction=float(actual.mean()),buckets=[dict(lower=float(lo),count=int(((prob>=lo)&(prob<lo+.2)).sum()),actual_error_fraction=float(actual[(prob>=lo)&(prob<lo+.2)].mean()) if ((prob>=lo)&(prob<lo+.2)).any() else None) for lo in [0,.2,.4,.6,.8]])
        calibration[m]['blend_weight_mean']=float(np.asarray(d['weight']).mean())
    CI={a:paired(error[a]-base,records) for a in ARMS if a!='WiLoR'};CI['spatial minus pooled']=paired(error['spatial']-error['pooled'],records)
    result=dict(complete=True,matched_right_frames=894,GT_points=truth.shape[0]*21,GT_inband_points=int(inband.sum()),GT_native_image_points=int(native_inimage.sum()),metrics=metrics,paired_sequence_CI=CI,risk_calibration=calibration,scope='native MANO amodal model projection; no independent visible2D labels; original annotation asset version unverified; six historically exposed clips; no full3D admission',production_admitted=False)
    (out/'metrics.json').write_text(json.dumps(result,indent=2));np.savez_compressed(out/'scoring_arrays.npz',GT_native_uv=truth,GT_canonical_uv=gtcrop,**pred)
    cards=[]
    for clip in sorted({r['clip'] for r in records}):
        ids=np.array([i for i,r in enumerate(records) if r['clip']==clip]);change=(error['spatial_blend']-base)[ids].mean(1);chosen=set()
        for label,order in [('best',np.argsort(change)),('worst',np.argsort(change)[::-1]),('still_hard',np.argsort(error['spatial_blend'][ids].mean(1))[::-1])]:
            i=next(int(ids[j]) for j in order if int(ids[j]) not in chosen);chosen.add(i);r=records[i];image=cv2.imread(r['image']);b=np.array(r['box']);center=(b[:2]+b[2:])/2;span=max(b[2:]-b[:2])*1.4;x0=max(0,int(center[0]-span/2));x1=min(1408,int(center[0]+span/2));y0=max(0,int(center[1]-span/2));y1=min(1408,int(center[1]+span/2));panels=[]
            for name,arm in [('RGB',None),('WiLoR self projection','WiLoR'),('independent RGB','spatial'),('risk-weighted blend','spatial_blend')]:
                drawn=image.copy()
                if arm:
                    for a,b in EDGES:cv2.line(drawn,tuple(np.rint(pred[arm][i,a]).astype(int)),tuple(np.rint(pred[arm][i,b]).astype(int)),(200,50,245),2,cv2.LINE_AA)
                    for pt in truth[i]:cv2.circle(drawn,tuple(np.rint(pt).astype(int)),3,(0,255,0),-1)
                crop=drawn[y0:y1,x0:x1];crop=cv2.resize(crop,None,fx=400/max(crop.shape[:2]),fy=400/max(crop.shape[:2]));panel=np.full((460,420,3),26,np.uint8);panel[50:50+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(panel,name,(8,27),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);panels.append(panel)
            file=f'{clip}_{label}_{r["frame"]}.jpg';cv2.imwrite(str(out/'images'/file),np.concatenate(panels,1));caption=f'{r["sequence"]} frame{r["frame"]} {label}: native image error WiLoR {base[i].mean():.2f}px, RGB {error["spatial"][i].mean():.2f}px, blend {error["spatial_blend"][i].mean():.2f}px. Green=amodal native MANO GT; purple=prediction.';cards.append(f'<figure><img src="images/{file}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    rows=''.join(f'<tr><td>{a}</td><td>{m["native_image_all21_px"]["mean"]:.2f}</td><td>{m["good_harmed_over1px"]}/{m["base_good5"]}</td><td>{m["recovered_to10"]}/{m["base_hard10"]}</td></tr>' for a,m in metrics.items())
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>独立RGB定位v9</title><style>body{background:#18212a;color:#eee;font:17px system-ui;max-width:1400px;margin:auto;padding:30px}td,th{border:1px solid #789;padding:12px}table{border-collapse:collapse}img{width:100%}a{color:#9df}</style><h1>独立RGB定位与风险：六录制894右手帧</h1><p>先封存两定位头及完整BA共八条输出再评分。本页为原生MANO投影的二维诊断，不是三维精度或可见性验收；六段有历史暴露。视觉主干冻结，只训定位/风险头，不称全端到端。</p><table><tr><th>方案</th><th>原生图像21点px</th><th>原≤5px点损伤&gt;1px</th><th>原&gt;10px恢复</th></tr>'+rows+'</table><p><a href="metrics.json">裁图/原图口径、风险校准、配对录制CI</a></p>'+''.join(cards))
    print(json.dumps(dict(metrics=metrics,CI=CI,calibration=calibration),indent=2))
if __name__=='__main__':main()
