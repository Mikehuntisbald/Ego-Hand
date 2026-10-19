"""Posthoc native-right depth audit; labels are opened only after final seal."""
import hashlib,json,sys,tarfile,time
from pathlib import Path
import cv2,numpy as np,smplx,torch
from scipy.optimize import linear_sum_assignment
from depth_common import PUBLIC,VIEW,build_views,select,hits
from geometry import load_frames,transform
sys.path.insert(0,'/mnt/why/native_mano_multiseq_v3_20261019/src')
from evaluate_validation import iou,stats
import wilor_eval_common as common
R=Path('/mnt/why/native_depth_audit_v8_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def signed(x):
    x=np.asarray(x);return dict(n=len(x),mean=float(np.mean(x)),median=float(np.median(x)),abs_mean=float(np.mean(np.abs(x))),abs_P95=float(np.percentile(np.abs(x),95)),abs_max=float(np.max(np.abs(x)))) if len(x) else dict(n=0)
def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    seal=json.loads((R/'predictions_frozen_r1.json').read_text());assert sha(R/'predictions_r1.json')==seal['predictions_sha256']
    rawseal=json.loads((R/'predictions_frozen.json').read_text());assert sha(R/'predictions.json')==rawseal['predictions_sha256']
    for name,h in rawseal['source_sha256'].items():assert sha(R/'src'/name)==h
    stereoseal=json.loads((R/'stereo_frozen.json').read_text())
    for name,h in stereoseal['files_sha256'].items():assert sha(R/'stereo_grid'/name)==h
    (R/'labels_open_order.json').write_text(json.dumps(dict(final_prediction_seal_sha256=sha(R/'predictions_frozen_r1.json'),label_open_unix=time.time(),note='New final predictions frozen before this grid scoring; frame0 and clip have previous diagnostic exposure'),indent=2))
    predictions=json.loads((R/'predictions_r1.json').read_text())['observations'];protocol=json.loads((R/'protocol.json').read_text())
    frames=load_frames(VIEW/'data/clip-002102');rows=json.loads((VIEW/'outputs/wilor_clip2102/observations.json').read_text())['frames'];surface=np.load(VIEW/'outputs/wilor_clip2102/surface.npz');rf=json.loads((VIEW/'outputs/rf_clip2102/predictions.json').read_text())['frames']
    layer=smplx.create(str(common.RUN/'assets/MANO_RIGHT.pkl'),'mano',use_pca=True,is_rhand=True,num_pca_comps=15).eval();results=[];unmatched=[];all_sensor=[];all_offset=[];alltotal=[]
    (R/'images').mkdir(exist_ok=False)
    with tarfile.open(PUBLIC/'assets/clip-002102-full.tar') as tar:
        beta=json.load(tar.extractfile('__hand_shapes.json__'))['mano']
        for fi in protocol['frames']:
            frame=frames[fi];frame['frame_index']=fi;ids=[i for i,r in enumerate(rows) if r['frame_index']==fi]
            hands=json.load(tar.extractfile(f'{fi:06d}.hands.json'));cal=json.load(tar.extractfile(f'{fi:06d}.cameras.json'))['214-1'];pose=transform(cal['T_world_from_camera']);targets=[]
            for side,h in hands.items():
                if '214-1' not in h.get('boxes_amodal',{}):continue
                box=np.array(h['boxes_amodal']['214-1']);box[[0,2]]=np.clip(box[[0,2]],0,1408);box[[1,3]]=np.clip(box[[1,3]],0,1408)
                if np.maximum(box[2:]-box[:2],0).prod()>=64:targets.append((side,box))
            overlaps=np.array([[iou(np.array(rows[i]['box_xyxy']),b) for side,b in targets] for i in ids]);a,b=linear_sum_assignment(1-overlaps)
            matched=[ids[x] for x,y in zip(a,b) if targets[y][0]=='right' and overlaps[x,y]>=.1]
            if len(matched)!=1:unmatched.append(fi);continue
            i=matched[0];h=hands['right']['mano_pose'];root=torch.tensor([h['wrist_xform']],dtype=torch.float32)
            with torch.no_grad():truth=layer(betas=torch.tensor([beta],dtype=torch.float32),global_orient=root[:,:3],hand_pose=torch.tensor([h['thetas']],dtype=torch.float32),transl=root[:,3:],return_verts=True)
            gt_mesh=((truth.vertices[0].numpy()-pose[:3,3])@pose[:3,:3]).astype(np.float32);gt_wrist=(truth.joints[0,0].numpy()-pose[:3,3])@pose[:3,:3]
            candidate=next(x for x in predictions if x['observation']==i);row=dict(frame_index=fi,observation=i,GT_wrist_camera_m=gt_wrist.tolist(),candidates=candidate['candidates'],decomposition=[])
            for name,c in row['candidates'].items():c['signed_wrist_Z_error_mm']=(c['depth_m']-float(gt_wrist[2]))*1000 if c['valid'] else None
            views=build_views(frame,ids,rows,R/'stereo_grid')
            with np.load(rf[fi]['output']) as v:masks=v['masks'];own=masks&~(masks.sum(0)>1)[None]
            image=cv2.imread(frame['images']['214-1']);assert image is not None;box=np.rint(rows[i]['box_xyxy']).astype(int);cv2.rectangle(image,tuple(box[:2]),tuple(box[2:]),(255,255,255),2)
            for vi,view in enumerate(views):
                points,loc,count=select(view,own[rows[i]['detection']]);direction=points/points[:,2:3];ph=hits(direction,surface['vertices'][i],surface['faces']);gh=hits(direction,gt_mesh,surface['faces'])
                offset=ph-surface['joints'][i,0,2];accepted=np.isfinite(ph)&(np.abs(offset)<.15);wrist=points[:,2]-offset;accepted&=(wrist>.08)&(wrist<3.);both=accepted&np.isfinite(gh)
                sensor=points[both,2]-gh[both];offset_error=(gh[both]-gt_wrist[2])-offset[both];total=wrist[both]-gt_wrist[2]
                assert np.max(np.abs(total-sensor-offset_error),initial=0)<1e-6
                all_sensor.extend(sensor*1000);all_offset.extend(offset_error*1000);alltotal.extend(total*1000)
                row['decomposition'].append(dict(target=view['target'],candidate_pixels=count,sampled=len(loc),accepted_old=int(accepted.sum()),also_GT_mesh_hits=int(both.sum()),stereo_surface_Z_signed_mm=signed(sensor*1000),surface_to_wrist_signed_mm=signed(offset_error*1000),total_signed_mm=signed(total*1000),confidence_matched=signed(view['confidence'].reshape(-1)[loc][both])))
                for k,position in enumerate(loc):
                    xy=(int(view['px'].reshape(-1)[position]),int(view['py'].reshape(-1)[position]));color=(80,80,80) if not both[k] else ((50,220,50) if abs((wrist[k]-gt_wrist[2])*1000)<10 else (20,20,245));cv2.circle(image,xy,2,color,-1)
            # Real source RGB; sample colors show diagnostic error, not visibility GT.
            image=cv2.resize(image,(704,704));canvas=np.full((850,1120,3),24,np.uint8);canvas[:704,:704]=image
            cv2.putText(canvas,f'Frame {fi}: real RGB / native MANO right GT',(20,735),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1)
            lines=[f'{name}: Z error {c["signed_wrist_Z_error_mm"]:+.1f}mm' if c['valid'] else f'{name}: invalid' for name,c in row['candidates'].items()]
            for k,line in enumerate(lines):cv2.putText(canvas,line,(715,80+70*k),cv2.FONT_HERSHEY_SIMPLEX,.46,(230,230,230),1)
            cv2.putText(canvas,'Samples: green abs error <10mm, red >=10mm, gray no shared GT surface',(20,780),cv2.FONT_HERSHEY_SIMPLEX,.52,(210,210,210),1)
            cv2.putText(canvas,'Diagnostic only: previously exposed clip; no labels used for these predictions',(20,815),cv2.FONT_HERSHEY_SIMPLEX,.52,(210,210,210),1)
            name=f'frame{fi:03d}.jpg';assert cv2.imwrite(str(R/'images'/name),canvas);row['image']=f'images/{name}';results.append(row)
    metrics={}
    for arm in protocol['variants']:
        valid=[r for r in results if r['candidates'][arm]['valid']];err=np.array([r['candidates'][arm]['signed_wrist_Z_error_mm'] for r in valid]);base=np.array([r['candidates']['baseline']['signed_wrist_Z_error_mm'] for r in valid]);good=np.abs(base)<=10;hard=np.abs(base)>20
        metrics[arm]=dict(signed_error_mm=signed(err),valid_right_frames=len(valid),fallback_frames=sum(bool(r['candidates'][arm].get('fallback_to_baseline')) for r in valid),good10=int(good.sum()),good_harmed_over1mm=int((good&(np.abs(err)>np.abs(base)+1)).sum()),hard20=int(hard.sum()),hard_recovered_to20=int((hard&(np.abs(err)<=20)).sum()),improved_over1mm=int((np.abs(err)<np.abs(base)-1).sum()),worsened_over1mm=int((np.abs(err)>np.abs(base)+1).sum()))
    output=dict(complete=True,posthoc_diagnostic_only=True,production_admitted=False,frames_requested=protocol['frames'],matched_right_frames=len(results),unmatched=unmatched,metrics=metrics,
      sampled_shared_ray_decomposition=dict(stereo_surface_Z_signed_mm=signed(all_sensor),surface_to_wrist_signed_mm=signed(all_offset),total_signed_mm=signed(alltotal)),
      scope='Nine right frames from one historically exposed clip; no independent CI or generalization claim. GT mesh uses existing RIGHT asset with original annotation parameters; original asset version not independently verified. Shared-ray conditional stats exclude uncovered rays and are not complete surface GT.',results=results)
    (R/'metrics.json').write_text(json.dumps(output,indent=2));print(json.dumps(dict(metrics=metrics,decomposition=output['sampled_shared_ray_decomposition']),indent=2),flush=True)
    table=''.join(f'<tr><td>{arm}</td><td>{v["valid_right_frames"]}</td><td>{v["signed_error_mm"]["mean"]:.2f}</td><td>{v["signed_error_mm"]["abs_mean"]:.2f}</td><td>{v["signed_error_mm"]["abs_max"]:.2f}</td><td>{v["improved_over1mm"]}/{v["worsened_over1mm"]}</td></tr>' for arm,v in metrics.items())
    figures=''.join(f'<figure><img src="{r["image"]}" width="900"><figcaption>frame {r["frame_index"]}</figcaption></figure>' for r in results)
    page='<!doctype html><meta charset="utf-8"><title>腕深度来源诊断v8</title><style>body{background:#18212a;color:#eee;font:17px system-ui;max-width:1180px;margin:auto;padding:30px}td,th{border:1px solid #789;padding:12px}table{border-collapse:collapse}a{color:#9df}</style><h1>腕深度来源诊断v8</h1><p>固定间隔9帧自然RGB，原LightESS不换权重；四组预测全部冻结再打开本轮标签。原方法逐项复现。仅右手原生MANO诊断，单一历史片段，不作新鲜泛化或生产准入。</p><table><tr><th>方案</th><th>有效右手帧</th><th>有符号Z误差mm</th><th>绝对Z均值mm</th><th>最大mm</th><th>改善/变差&gt;1mm帧</th></tr>'+table+'</table><p>erode4：原生mask内缩4像素。iter5：保持形状/姿态，沿原预测腕射线平移网格，最多五次表面校正。无效候选按预声明回退原锚，raw结果保留。</p><p>射线分解仅在预测与GT网格均相交的旧接受样本上，覆盖数量单列；不能解释为独立真实表面深度标注。GT网格只做事后诊断，未进入预测，也未用于校准偏置。</p><p><a href="metrics.json">逐帧统计和覆盖/分解</a> · <a href="protocol.json">预声明规则</a> · <a href="predictions_frozen_r1.json">预测封存</a> · <a href="labels_open_order.json">标签打开顺序</a></p>'+figures
    (R/'report.html').write_text(page)
if __name__=='__main__':main()
