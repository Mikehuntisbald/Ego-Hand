"""Post-freeze native21 right-hand development evaluation; natural RGB cases."""
import collections,hashlib,html,json,sys,tarfile,time
from pathlib import Path
import cv2,numpy as np,torch,smplx
from scipy.optimize import linear_sum_assignment
R=Path('/mnt/why/native_mano_supervision_v5_20261019');V=Path('/mnt/why/native_mano_multiseq_v3_20261019')
sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src');sys.path.insert(0,str(V/'src'));sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/src')
from native_mano import NativeMANO,common,exp_so3
from geometry import transform,camera
from evaluate_validation import iou,stats,protection,clustered_ci
NEW=[a+s for a in ['native_regression','proxy_dit','native_dit','native_protected'] for s in ['', '_guard']]
OLD=['WiLoR','cosine_protected_displacement','rollout_dit_displacement']
ARMS=OLD+NEW
EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rel(x):return x-x[:,:1]
def paired(delta,records,cluster):
    groups=sorted({r[cluster] for r in records});sums=np.array([delta[[r[cluster]==g for r in records]].sum() for g in groups]);counts=np.array([delta[[r[cluster]==g for r in records]].size for g in groups])
    rng=np.random.default_rng(190019);draw=rng.integers(len(groups),size=(10000,len(groups)));means=sums[draw].sum(1)/counts[draw].sum(1)
    return dict(clusters=len(groups),point_weighted_delta_mm=float(delta.mean()),CI95_mm=np.quantile(means,[.025,.975]).tolist(),bootstrap_draws=10000,units='named candidate minus reference mm; negative better',limitation='six historically exposed development recordings; only three people')

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    frozen=json.loads((R/'predictions_frozen.json').read_text());assert frozen['complete']
    for name,value in frozen['seals'].items():
        seal=R/name;assert sha(seal)==value
        for file,digest in json.loads(seal.read_text())['files'].items():assert sha(seal.parent/file)==digest
    out=R/'evaluation';out.mkdir(exist_ok=False);(out/'images').mkdir()
    (out/'label_open_receipt.json').write_text(json.dumps(dict(unix=time.time(),predictions_frozen_sha256=sha(R/'predictions_frozen.json'),historical_evaluation_before_v5_design=True),indent=2))
    protocol=json.loads((V/'protocol.json').read_text());records=[];pred={a:[] for a in ARMS};counts=collections.Counter();oldUM=np.load(V/'evaluation_r3/labels_and_errors.npz')
    for clip in protocol['clips']:
        view=Path(clip['view']);rows=json.loads((view/'outputs/wilor_clip2102/observations.json').read_text())['frames'];arrays={}
        for a in OLD:arrays[a]=np.load(view/('predictions_rollout' if a.startswith('rollout_') else 'predictions')/f'{a}.npz')['joints_camera21']
        for a in NEW:arrays[a]=np.load(R/'predictions'/f"clip-{clip['clip']:06}"/f'{a}.npz')['joints_camera21']
        with tarfile.open(clip['archive']) as tar:
            beta=json.load(tar.extractfile('__hand_shapes.json__'))['mano']
            for fi in range(150):
                hands=json.load(tar.extractfile(f'{fi:06}.hands.json'));cal=json.load(tar.extractfile(f'{fi:06}.cameras.json'))['214-1'];pose=transform(cal['T_world_from_camera']);targets=[]
                for side,h in hands.items():
                    if side=='right':counts['annotated_right_hand_frames']+=1
                    if not h.get('umetrack_pose') or '214-1' not in h.get('boxes_amodal',{}):
                        if side=='right':counts['right_without_matching_box_or_UM_pose']+=1
                        continue
                    box=np.array(h['boxes_amodal']['214-1']);box[[0,2]]=np.clip(box[[0,2]],0,1408);box[[1,3]]=np.clip(box[[1,3]],0,1408)
                    if np.maximum(box[2:]-box[:2],0).prod()>=64:targets.append((side,box))
                ids=[i for i,r in enumerate(rows) if r['frame_index']==fi]
                if not ids or not targets:continue
                overlap=np.array([[iou(np.array(rows[i]['box_xyxy']),t[1]) for t in targets] for i in ids]);a,b=linear_sum_assignment(1-overlap)
                for x,y in zip(a,b):
                    side=targets[y][0]
                    if overlap[x,y]<.1 or side!='right' or not hands[side].get('mano_pose'):continue
                    i=ids[x];h=hands[side]['mano_pose'];counts['matched_right_hand_frames']+=1;counts['matched_predicted_right']+=int(rows[i]['right'])
                    records.append(dict(clip=clip['clip'],sequence=clip['sequence'],subject=clip['subject'],observation=i,image=rows[i]['image'],box=rows[i]['box_xyxy'],frame=fi,track=rows[i]['track_id'],timestamp_ns=rows[i]['timestamp_ns'],
                        calibration=cal['calibration'],theta=h['thetas'],wrist_xform=h['wrist_xform'],beta=beta,R=pose[:3,:3],T=pose[:3,3],UM=oldUM[f"clip{clip['clip']}_GT_camera20"][i]))
                    for arm in ARMS:pred[arm].append(arrays[arm][i])
    assert len(records)==894,'Same matched right subset as predeclared geometry audit'
    asset=common.RUN/'assets/MANO_RIGHT.pkl';layer=smplx.create(str(asset),'mano',use_pca=True,is_rhand=True,num_pca_comps=15).eval();codec=NativeMANO(device='cpu')
    beta=torch.tensor(np.asarray([r['beta'] for r in records]),dtype=torch.float32);theta=torch.tensor(np.asarray([r['theta'] for r in records]),dtype=torch.float32);root=torch.tensor(np.asarray([r['wrist_xform'] for r in records]),dtype=torch.float32)
    with torch.no_grad():
        gt=layer(betas=beta,global_orient=root[:,:3],hand_pose=theta,transl=root[:,3:],return_verts=True,return_full_pose=True)
        xyz,vertices=codec.decode(gt.joints[:,0],exp_so3(gt.full_pose[:,:3]),exp_so3(gt.full_pose[:,3:].reshape(-1,15,3)),beta,torch.ones(len(records),dtype=torch.bool),vertices=True)
        parity=float((vertices-gt.vertices).abs().max());assert parity<1e-6
    Rs=np.asarray([r['R'] for r in records]);Ts=np.asarray([r['T'] for r in records]);GT=np.einsum('nji,nkj->nki',Rs,xyz.numpy()-Ts[:,None]);UM=np.asarray([r['UM'] for r in records]);pred={a:np.asarray(x) for a,x in pred.items()}
    absolute={a:np.linalg.norm(x-GT,axis=-1)*1000 for a,x in pred.items()};relative={a:np.linalg.norm(rel(x)-rel(GT),axis=-1)[:,1:]*1000 for a,x in pred.items()};valid=np.ones_like(relative['WiLoR'],bool)
    def summarise(mask):
        output={}
        for a in ARMS:
            mapped=pred[a][:,common.MAPPING];errorUM=np.linalg.norm((mapped-mapped[:,5:6])-(UM-UM[:,5:6]),axis=-1)[:,np.arange(20)!=5]*1000
            output[a]=dict(camera21_mm=stats(absolute[a][mask].ravel()),native_relative20_mm=stats(relative[a][mask].ravel()),wrist_mm=stats(absolute[a][mask,0]),
                native_relative_preservation=protection(relative['WiLoR'][mask],relative[a][mask],valid[mask]),UM_aux_relative19_mm=stats(errorUM[mask].ravel()))
        return output
    allmask=np.ones(len(records),bool);strict=np.array([r['subject'] in ['P0010','P0015'] for r in records]);metrics=summarise(allmask);strictmetrics=summarise(strict)
    perclip={str(clip):summarise(np.array([r['clip']==clip for r in records])) for clip in sorted({r['clip'] for r in records})}
    CIs={}
    for candidate,reference in [(a,'WiLoR') for a in ARMS if a!='WiLoR']+[('native_dit','proxy_dit'),('native_dit_guard','proxy_dit_guard'),('native_protected','native_dit'),('native_protected_guard','native_dit_guard')]:
        delta=relative[candidate]-relative[reference];CIs[candidate+' minus '+reference]={kind:paired(delta,records,kind) for kind in ['sequence','subject']}
    motion={a:collections.defaultdict(list) for a in ARMS};groups=collections.defaultdict(list)
    for i,r in enumerate(records):groups[(r['sequence'],r['clip'],r['track'])].append(i)
    for key,ids in groups.items():
        ids=np.asarray(sorted(ids,key=lambda i:records[i]['timestamp_ns']));dt=np.diff(np.array([records[i]['timestamp_ns'] for i in ids],np.int64))/1e9
        if len(ids)<3:continue
        pair=(dt>0)&(dt<.1);triple=pair[:-1]&pair[1:];world=xyz.numpy()[ids];g=rel(world);vg=np.diff(g,axis=0)/(60*dt[:,None,None]);ag=np.diff(vg,axis=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2)
        speed=np.linalg.norm(vg,axis=-1)[:,1:];fast=(speed>=np.quantile(speed[pair],.9))&pair[:,None]
        for a in ARMS:
            p=np.einsum('nij,nkj->nki',Rs[ids],pred[a][ids])+Ts[ids,None];rp=rel(p);vp=np.diff(rp,axis=0)/(60*dt[:,None,None]);ap=np.diff(vp,axis=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2);m=motion[a]
            e=np.linalg.norm(vp-vg,axis=-1)[:,1:]*1000;m['relative_velocity'].extend(e[pair].ravel());m['relative_acceleration'].extend((np.linalg.norm(ap-ag,axis=-1)[:,1:]*1000)[triple].ravel())
            m['fast_pred'].extend(np.linalg.norm(vp,axis=-1)[:,1:][fast]);m['fast_GT'].extend(speed[fast]);m['velocity_error_over10'].extend((e>10)[pair].ravel())
            lengths=np.stack([np.linalg.norm(pred[a][ids,x]-pred[a][ids,y],axis=-1) for x,y in EDGES],-1);m['bone_cv'].extend(lengths.std(0)/np.maximum(lengths.mean(0),1e-6))
    temporal={a:dict(relative_velocity_error_mm_at60=stats(m['relative_velocity']),relative_acceleration_error_mm_at60sq=stats(m['relative_acceleration']),relative_velocity_error_over10=int(np.sum(m['velocity_error_over10'])),
        fast_relative_motion_amplitude_ratio=float(np.mean(m['fast_pred'])/np.mean(m['fast_GT'])),bone_length_CV_mean=float(np.mean(m['bone_cv']))) for a,m in motion.items()}
    summary=dict(complete=True,right_only=True,counts=dict(counts),metrics=metrics,per_clip=perclip,paired_CI=CIs,temporal=temporal,strict_optimizer_selection_unseen_subjects=dict(subjects=['P0010','P0015'],hand_frames=int(strict.sum()),metrics=strictmetrics),
        source_asset_sha256=sha(asset),official_mesh_parity_max_m=parity,scope='native MANO21 decoded through existing licensed RIGHT asset using WiLoR exact joint/tip semantics; annotation-original asset not independently verified',
        validation_exposure='six historically evaluated development recordings; P10/P15 labels excluded from optimizer and checkpoint selection but already audited before v5 design',
        full_pipeline=False,prediction_freeze_sha256=sha(R/'predictions_frozen.json'),production_admitted=False,ground_truth_inference=False,left_hand='passthrough coarse WiLoR; not trained or claimed improved')
    (out/'metrics.json').write_text(json.dumps(summary,indent=2));np.savez_compressed(out/'sealed_scoring_arrays.npz',GT_native_camera21=GT,**{a:pred[a] for a in ARMS},GT_UM_aux20=UM)
    cards=[];candidate='native_protected_guard'
    for clip in sorted({r['clip'] for r in records}):
        ids=np.array([i for i,r in enumerate(records) if r['clip']==clip]);change=relative[candidate][ids].mean(1)-relative['WiLoR'][ids].mean(1);hard=relative[candidate][ids].mean(1);chosen=set()
        for category,order in [('best_change',np.argsort(change)),('worst_change',np.argsort(change)[::-1]),('still_hard',np.argsort(hard)[::-1])]:
            i=next(int(ids[x]) for x in order if int(ids[x]) not in chosen);chosen.add(i);r=records[i];photo=cv2.imread(r['image']);assert photo is not None
            b=np.array(r['box']);c=(b[:2]+b[2:])/2;s=max(b[2:]-b[:2])*1.4;x0=max(0,int(c[0]-s/2));x1=min(1408,int(c[0]+s/2));y0=max(0,int(c[1]-s/2));y1=min(1408,int(c[1]+s/2));cam=camera(r['calibration']);truth=cam.eye_to_window(GT[i]);panels=[]
            for name,arm in [('RGB',None),('WiLoR','WiLoR'),('matched proxy DiT','proxy_dit_guard'),('native protected DiT',candidate)]:
                image=photo.copy()
                if arm:
                    uv=cam.eye_to_window(pred[arm][i]);color=(190,50,250)
                    for a,b in EDGES:cv2.line(image,tuple(np.rint(uv[a]).astype(int)),tuple(np.rint(uv[b]).astype(int)),color,2,cv2.LINE_AA)
                    for pt in truth:cv2.circle(image,tuple(np.rint(pt).astype(int)),3,(0,255,0),-1)
                crop=image[y0:y1,x0:x1];crop=cv2.resize(crop,None,fx=400/max(crop.shape[:2]),fy=400/max(crop.shape[:2]));panel=np.full((460,420,3),26,np.uint8);panel[50:50+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(panel,name,(8,27),cv2.FONT_HERSHEY_SIMPLEX,.51,(255,255,255),1);panels.append(panel)
            file=f"{clip}_{category}_{r['frame']}.jpg";cv2.imwrite(str(out/'images'/file),np.concatenate(panels,1))
            caption=f"{r['sequence']} frame {r['frame']} {category}: native20 relative {relative['WiLoR'][i].mean():.2f} → {relative[candidate][i].mean():.2f} mm. Green=native MANO GT; purple=prediction."
            cards.append(f'<figure><img src="images/{file}"><figcaption>{html.escape(caption)}</figcaption></figure>')
    table=''
    for arm,m in metrics.items():
        p=m['native_relative_preservation'];table+=f"<tr><td>{arm}</td><td>{m['native_relative20_mm']['mean']:.2f}</td><td>{m['camera21_mm']['mean']:.2f}</td><td>{p['good_harmed_over1mm']}/{p['base_good10']}</td><td>{p['good_to_over20']}</td><td>{p['recovered_to20']}/{p['base_hard20']}</td></tr>"
    page='''<!doctype html><meta charset="utf-8"><title>原生MANO监督v5</title><style>body{font:17px system-ui;max-width:1400px;margin:32px auto;background:#161c22;color:#eee;padding:20px}td,th{border:1px solid #78828c;padding:10px}table{border-collapse:collapse}img{width:100%}a{color:#7fd4ff}figcaption{margin:12px}pre{white-space:pre-wrap}</style><h1>原生MANO参数监督：右手受控试验</h1>
    <p>六名训练人员155窗口、开发选择25窗口。相同样本/初始化/600更新，直接回归、旧伪目标DiT、原生参数DiT及原生准确点保护。原生目标不经过UmeTrack拟合，保持腕位置和预测形状。视觉主干冻结、空间适配/时序/DiT可训练。推理保留自然RGB与17时隙，中心参数4采样/10步均值，不声称整轨迹联合扩散。</p>
    <p>仅右手：尚缺可核验LEFT资产，左手按预测手侧原样通过。GT由数据集MANO参数和现有RIGHT资产解码，21点/指尖采用WiLoR约定；原标注资产版本尚未独立核验。六录制894匹配右手帧，历史开发数据已经查看，不作新鲜盲测。P10/P15只表示未进入优化器或权重选择。</p>
    <p>guard沿用5/10/30mm的参数位移限制，未放宽；它限制相对原预测的变化，不能保证GT准确。当前表是BA前补全模块；不冒充完整headpose/深度/BA或生产准入。</p>
    <table><tr><th>方案</th><th>原生20点腕对齐mm</th><th>相机21点mm</th><th>原≤10mm点损伤>1mm</th><th>原≤10→>20mm</th><th>原>20mm恢复到≤20mm</th></tr>'''+table+'</table><p><a href="metrics.json">完整数值/区间/真实时间运动/形状连贯指标</a></p>'+''.join(cards)
    (out/'report.html').write_text(page);print(json.dumps({a:dict(relative=m['native_relative20_mm']['mean'],camera=m['camera21_mm']['mean'],preservation=m['native_relative_preservation']) for a,m in metrics.items()},indent=2))

if __name__=='__main__':main()
