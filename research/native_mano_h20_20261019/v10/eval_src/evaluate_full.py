"""Native right3D, whole output consistency and auxiliary left/both UM audit."""
import collections,hashlib,json,sys,tarfile,time
from pathlib import Path
import cv2,h5py,numpy as np,smplx,torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
R=Path('/mnt/why/native_candidate_sequence_v10_20261019');VIEW=R/'full_pipeline';V7=Path('/mnt/why/native_mano_pool_v7_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019')
sys.path.insert(0,str(V2/'src'));sys.path.insert(0,'/mnt/why/native_mano_multiseq_v3_20261019/src');sys.path.insert(0,str(PUBLIC/'src'))
from native_mano import NativeMANO,common,exp_so3
from geometry import transform,camera,load_frames
from evaluate_validation import iou,stats,protection
from hand_tracking_toolkit.dataset import decode_hand_pose
from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json,forward_kinematics
ARMS=['frame_RGB','temporal_noRGB','temporal_RGB','temporal_noRGB_scaled','temporal_RGB_scaled'];EDGES=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);assert json.loads((R/'full_status.json').read_text())['complete'];frozen=json.loads((VIEW/'receipts/predictions_frozen.json').read_text());files={}
    for name,h in frozen['seals'].items():
        seal=VIEW/name;assert sha(seal)==h
        for n,digest in json.loads(seal.read_text())['files'].items():assert sha(seal.parent/n)==digest
    for a in ARMS:
        for suffix in ['', '_guarded']:files[a+suffix]=VIEW/f'outputs/ba_{a}{suffix}/parameters.npz'
    files.update(BA=VIEW/'outputs/mano_baseline_clip2102/parameters.npz',old_v5_guard=Path('/mnt/why/native_mano_supervision_v5_20261019/full_pipeline/outputs/ba_rollout_dit_guarded/parameters.npz'),v7_DiT=V7/'full_pipeline/outputs/ba_full/parameters.npz',v7_DiT_guard=V7/'full_pipeline/outputs/ba_full_guarded/parameters.npz',v9_RGB_DiT_guard=Path('/mnt/why/native_rgb_observation_v9_20261019/full_pipeline/outputs/ba_DiT_spatial_guarded/parameters.npz'))
    data={k:np.load(p) for k,p in files.items()};base=data['BA'];rows=json.loads((VIEW/'outputs/wilor_clip2102/observations.json').read_text())['frames'];frames=load_frames(VIEW/'data/clip-002102');head=json.loads((VIEW/'outputs/head_clip2102/headpose.json').read_text())['frames'];codec=NativeMANO(device='cpu');fixed={k:v.clone() for k,v in codec.state_dict().items()};output_checks={}
    tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32)
    for a,p in files.items():
        v=data[a]
        for k in ['timestamp_ns','track_id','right','group','frame_index']:assert np.array_equal(v[k],base[k]),(a,k)
        with torch.no_grad():back=codec.from_state(tensor(v['state_world51']),tensor(v['betas_shared'][v['group']]),torch.tensor(v['right']))
        parity=float(np.abs(back.numpy()-v['joints_world']).max());assert parity<1e-6,(a,parity)
        if a not in ARMS and a not in [x+'_guarded' for x in ARMS]:continue
        payload=json.loads((p.parent/'joints3d_world.json').read_text());assert len(payload['frames'])==len(frames)==150
        with h5py.File(p.parent/'bundle.h5') as h5:
            assert np.array_equal(h5['timestamp_ns'][:],np.array([f['timestamp_ns'] for f in frames],np.int64))
            for tid in np.unique(v['track_id']):
                ids=np.flatnonzero(v['track_id']==tid);g=h5[f'track_{tid}'];valid=g['valid'][:];assert np.array_equal(np.flatnonzero(valid),v['frame_index'][ids]);assert np.allclose(g['joints_world_m'][:][v['frame_index'][ids]],v['joints_world'][ids],atol=1e-7)
        for i,row in enumerate(rows):
            fi=int(v['frame_index'][i]);hand=next(x for x in payload['frames'][fi]['hands'] if x['track_id']==int(v['track_id'][i]));assert hand['valid'] and hand['right']==bool(v['right'][i]);assert np.allclose(hand['joints_world_m'],v['joints_world'][i],atol=1e-7);assert np.allclose(hand['joints_camera_m'],v['joints_camera'][i],atol=1e-7);assert np.allclose(hand['local_axis_angle_rad'],v['state_world51'][i,6:].reshape(15,3),atol=1e-7);assert np.allclose(Rotation.from_quat(hand['root_quaternion_xyzw']).as_matrix(),Rotation.from_rotvec(v['state_world51'][i,3:6]).as_matrix(),atol=1e-7)
        output_checks[a]=dict(complete=True,FK_max_m=parity,JSON_H5_and_timeaxis=True,preserved_identity_chirality=True)
    assert all(torch.equal(v,fixed[k]) for k,v in codec.state_dict().items());(R/'receipts/full_output_verification.json').write_text(json.dumps(dict(complete=True,outputs=output_checks,fixed_FK_buffers=True),indent=2))
    (R/'receipts/full_label_open_order.json').write_text(json.dumps(dict(unix=time.time(),all_ten_prediction_seal_sha256=sha(VIEW/'receipts/predictions_frozen.json'),historical_exposure=True),indent=2))
    beta=[];theta=[];root=[];rot=[];trans=[];ids=[];calibrations=[];gtUM=np.full((len(rows),20,3),np.nan);sides=np.full(len(rows),'',dtype='<U5');reference_head=[]
    with tarfile.open(PUBLIC/'assets/clip-002102-full.tar') as tar:
        shapes=json.load(tar.extractfile('__hand_shapes.json__'));shape=from_json(shapes['umetrack']);native_shape=shapes['mano']
        for fi in range(150):
            h=json.load(tar.extractfile(f'{fi:06}.hands.json'));cal=json.load(tar.extractfile(f'{fi:06}.cameras.json'))['214-1'];pose=transform(cal['T_world_from_camera']);reference_head.append(pose);targets=[];poses=decode_hand_pose(h)
            for side,x in h.items():
                if '214-1' not in x.get('boxes_amodal',{}):continue
                box=np.array(x['boxes_amodal']['214-1']);box[[0,2]]=np.clip(box[[0,2]],0,1408);box[[1,3]]=np.clip(box[[1,3]],0,1408)
                if np.maximum(box[2:]-box[:2],0).prod()>=64:targets.append((side,box))
            obs=[i for i,r in enumerate(rows) if r['frame_index']==fi];overlap=np.array([[iou(np.array(rows[i]['box_xyxy']),t[1]) for t in targets] for i in obs]);a,b=linear_sum_assignment(1-overlap)
            for x,y in zip(a,b):
                if overlap[x,y]<.1:continue
                i=obs[x];side=targets[y][0];sides[i]=side;up=next(p for s,p in poses.items() if s.value==side)
                if up.umetrack is not None:
                    with torch.no_grad():world=forward_kinematics(up.umetrack,shape,requires_mesh=False)[0].numpy()
                    gtUM[i]=(world-pose[:3,3])@pose[:3,:3]
                if side!='right' or not h['right'].get('mano_pose'):continue
                target=h['right']['mano_pose'];ids.append(i);beta.append(native_shape);theta.append(target['thetas']);root.append(target['wrist_xform']);rot.append(pose[:3,:3]);trans.append(pose[:3,3]);calibrations.append(cal['calibration'])
    assert len(ids)==150;layer=smplx.create(str(common.RUN/'assets/MANO_RIGHT.pkl'),'mano',use_pca=True,is_rhand=True,num_pca_comps=15).eval();root=tensor(root);beta=tensor(beta)
    with torch.no_grad():
        o=layer(betas=beta,global_orient=root[:,:3],hand_pose=tensor(theta),transl=root[:,3:],return_verts=True,return_full_pose=True);gtworld=codec.decode(o.joints[:,0],exp_so3(o.full_pose[:,:3]),exp_so3(o.full_pose[:,3:].reshape(-1,15,3)),beta,torch.ones(len(ids),dtype=torch.bool)).numpy()
    GT=np.einsum('nji,nkj->nki',np.asarray(rot),gtworld-np.asarray(trans)[:,None]);pred={a:v['joints_camera'][ids] for a,v in data.items()};rel=lambda x:x-x[:,:1];error={a:np.linalg.norm(p-GT,axis=-1)*1000 for a,p in pred.items()};relative={a:np.linalg.norm(rel(p)-rel(GT),axis=-1)[:,1:]*1000 for a,p in pred.items()};gauge=reference_head[0]@np.linalg.inv(np.array(head[0]['T_world_from_rgb']));t=base['timestamp_ns'][ids];dt=np.diff(t)/1e9;pair=(dt>0)&(dt<.1);triple=pair[:-1]&pair[1:];g=rel(gtworld);vg=np.diff(g,axis=0)/(60*dt[:,None,None]);ag=np.diff(vg,axis=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2);speed=np.linalg.norm(vg,axis=-1)[:,1:];fast=(speed>=np.quantile(speed[pair],.9))&pair[:,None]
    metrics={};UM={};native_valid=np.ones_like(relative['BA'],bool);absolute_valid=np.ones_like(error['BA'],bool);validUM=np.isfinite(gtUM).all(-1)
    baseUM=np.linalg.norm(base['joints_camera'][:,common.MAPPING]-gtUM,axis=-1)*1000
    for a,p in pred.items():
        world=data[a]['joints_world'][ids]@gauge[:3,:3].T+gauge[:3,3];rp=rel(world);vp=np.diff(rp,axis=0)/(60*dt[:,None,None]);ap=np.diff(vp,axis=0)/(60*(dt[1:]+dt[:-1])[:,None,None]/2);velocity=np.linalg.norm(vp-vg,axis=-1)[:,1:]*1000;lengths=np.stack([np.linalg.norm(p[:,x]-p[:,y],axis=-1) for x,y in EDGES],-1)
        metrics[a]=dict(camera21_mm=stats(error[a].ravel()),relative20_mm=stats(relative[a].ravel()),wrist_mm=stats(error[a][:,0]),world_firstpose_SE3_mm=stats((np.linalg.norm(world-gtworld,axis=-1)*1000).ravel()),relative_preservation=protection(relative['BA'],relative[a],native_valid),absolute_preservation=protection(error['BA'],error[a],absolute_valid),relative_velocity_error_mm_at60=stats(velocity[pair].ravel()),relative_acceleration_error_mm_at60sq=stats((np.linalg.norm(ap-ag,axis=-1)[:,1:]*1000)[triple].ravel()),relative_velocity_error_over10=int((velocity[pair]>10).sum()),fast_relative_motion_amplitude_ratio=float(np.linalg.norm(vp,axis=-1)[:,1:][fast].mean()/speed[fast].mean()),bone_length_CV_mean=float((lengths.std(0)/lengths.mean(0).clip(1e-6)).mean()),wrist_Z_signed_mean_mm=float((p[:,0,2]-GT[:,0,2]).mean()*1000),relative_preservation_vs_v9=protection(relative['v9_RGB_DiT_guard'],relative[a],native_valid),absolute_preservation_vs_v9=protection(error['v9_RGB_DiT_guard'],error[a],absolute_valid))
        e=np.linalg.norm(data[a]['joints_camera'][:,common.MAPPING]-gtUM,axis=-1)*1000;UM[a]={}
        for side,mask in [('both',validUM),('right',validUM&(sides=='right')[:,None]),('left',validUM&(sides=='left')[:,None])]:UM[a][side]=dict(camera_mapped20_mm=stats(e[mask]),preservation=protection(baseUM,e,mask))
    report=dict(complete=True,native_right_frames=150,metrics=metrics,auxiliary_UM20=UM,scope='Native MANO21 right hand through existing RIGHT asset; original annotation asset version unverified. UM20 is cross-model diagnostic only, separately names left/right. One historically exposed clip: no sequence CI.',all_new_predictions_frozen_before_labels=True,GT_inference=False,production_admitted=False)
    (R/'receipts/full_native_metrics.json').write_text(json.dumps(report,indent=2));np.savez_compressed(R/'receipts/full_scoring_arrays.npz',GT_native_camera21=GT,observation_indices=np.array(ids),GT_UM20=gtUM,**pred)
    out=R/'full_review';out.mkdir(exist_ok=False);(out/'images').mkdir();candidate='temporal_RGB_scaled_guarded';change=relative[candidate].mean(1)-relative['v9_RGB_DiT_guard'].mean(1);chosen=set();cards=[]
    for label,order in [('best',np.argsort(change)),('worst',np.argsort(change)[::-1]),('still_hard',np.argsort(error[candidate].mean(1))[::-1])]:
        j=next(int(x) for x in order if int(x) not in chosen);chosen.add(j);i=ids[j];row=rows[i];photo=cv2.imread(row['image']);box=np.array(row['box_xyxy']);center=(box[:2]+box[2:])/2;span=max(box[2:]-box[:2])*1.5;x0=max(0,int(center[0]-span/2));x1=min(1408,int(center[0]+span/2));y0=max(0,int(center[1]-span/2));y1=min(1408,int(center[1]+span/2));cam=camera(calibrations[j]);truth=cam.eye_to_window(GT[j]);panels=[]
        for name,arm in [('RGB',None),('base MANO BA','BA'),('native DiT + RGB guard','v9_RGB_DiT_guard'),('whole track selection',candidate)]:
            image=photo.copy()
            if arm:
                uv=cam.eye_to_window(pred[arm][j])
                for x,y in EDGES:cv2.line(image,tuple(np.rint(uv[x]).astype(int)),tuple(np.rint(uv[y]).astype(int)),(190,50,250),2,cv2.LINE_AA)
                for point in truth:cv2.circle(image,tuple(np.rint(point).astype(int)),3,(0,255,0),-1)
            crop=image[y0:y1,x0:x1];crop=cv2.resize(crop,None,fx=400/max(crop.shape[:2]),fy=400/max(crop.shape[:2]));panel=np.full((480,420,3),26,np.uint8);panel[50:50+crop.shape[0],:crop.shape[1]]=crop;cv2.putText(panel,name,(8,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
            if arm:cv2.putText(panel,f'3D {error[arm][j].mean():.2f} / rel {relative[arm][j].mean():.2f} mm',(8,470),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
            panels.append(panel)
        name=f'{label}_frame{row["frame_index"]}.jpg';cv2.imwrite(str(out/'images'/name),np.concatenate(panels,1));cards.append(f'<figure><img src="images/{name}"><figcaption>{label}, frame{row["frame_index"]}; green native MANO GT, purple prediction. 3D and relative errors shown separately.</figcaption></figure>')
    table=''
    for a,m in metrics.items():
        p=m['relative_preservation'];table+=f'<tr><td>{a}</td><td>{m["camera21_mm"]["mean"]:.3f}</td><td>{m["relative20_mm"]["mean"]:.3f}</td><td>{m["wrist_mm"]["mean"]:.3f}</td><td>{p["good_harmed_over1mm"]}/{p["base_good10"]}</td><td>{p["recovered_to20"]}/{p["base_hard20"]}</td></tr>'
    (out/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>原生MANO候选整段BA v10</title><style>body{background:#18212a;color:#eee;font:17px system-ui;max-width:1450px;margin:auto;padding:30px}td,th{border:1px solid #789;padding:10px}table{border-collapse:collapse}img{width:100%}a{color:#9df}</style><h1>原生MANO候选整段选择：完整回放</h1><p>固定RF/WiLoR、预测相机、物理深度与BA所有权重；无图像观测适配器对照逐值复现原v7 BA。新五组原始/保护共十条输出全部封存再评分。仅150右手帧原生GT诊断，单历史片段不作新鲜泛化。保护线5/10/30mm和原WiLoR参考重投影+0.2px不变。</p><table><tr><th>方案</th><th>相机21点mm</th><th>腕对齐20点mm</th><th>腕mm</th><th>原准确点损伤&gt;1mm</th><th>困难点恢复</th></tr>'+table+'</table><p>表中保护统计以原生相对误差为准，完整绝对保护、真实时间运动/骨长/快速动作和左右手UM辅助口径见根目录receipts/full_native_metrics.json。二维风险不代表3D可信度，BA仍无严格碰撞/关节限位保证。</p>'+''.join(cards))
    print(json.dumps({a:dict(camera=m['camera21_mm']['mean'],relative=m['relative20_mm']['mean'],wrist=m['wrist_mm']['mean'],preservation=m['relative_preservation']) for a,m in metrics.items()},indent=2))
if __name__=='__main__':main()
