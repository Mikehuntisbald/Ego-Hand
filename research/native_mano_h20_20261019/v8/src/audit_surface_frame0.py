"""Posthoc right-hand decomposition of cached stereo and surface correction.

GT mesh is diagnostic only. It never creates inference depth or replaces an
anchor. Preserve the exact two existing disparity outputs and sampling rule.
"""
import hashlib,json,sys,tarfile
from pathlib import Path
import numpy as np,torch,smplx
from scipy.optimize import linear_sum_assignment
R=Path('/mnt/why/native_depth_audit_v8_20261019');VIEW=Path('/mnt/why/native_mano_dit_v2_20261019/coherent_frontend');PUBLIC=Path('/mnt/why/public_headhand_h20_20261019')
sys.path.insert(0,str(PUBLIC/'src'));sys.path.insert(0,str(PUBLIC/'vendor/Ego-Hand/research'));sys.path.insert(0,'/mnt/why/native_mano_multiseq_v3_20261019/src')
from geometry import load_frames,camera,rectified_pair,remap_geometry,transform
from wilor_eval_common import unproject
import wilor_eval_common as common
from evaluate_validation import iou,stats

def surface_intersections(directions,vertices,faces):
    # Exact existing run_depth.py two-sided Moller-Trumbore convention (Z=1).
    triangles=vertices[faces];v0=triangles[:,0];edge1,edge2=triangles[:,1]-v0,triangles[:,2]-v0
    p=np.cross(directions[:,None],edge2[None]);determinant=(p*edge1[None]).sum(-1)
    inverse=np.divide(1.,determinant,out=np.zeros_like(determinant),where=np.abs(determinant)>1e-9)
    u=(p*(-v0)[None]).sum(-1)*inverse;q=np.cross(-v0,edge1);v=(directions[:,None]*q[None]).sum(-1)*inverse
    distance=(edge2*q).sum(-1)[None]*inverse;valid=(np.abs(determinant)>1e-9)&(u>=0)&(v>=0)&(u+v<=1)&(distance>.01)
    return np.where(valid,distance,np.inf).min(-1)

def main():
    torch.set_num_threads(4);R.mkdir(exist_ok=True);frame=load_frames(VIEW/'data/clip-002102')[0]
    rows=json.loads((VIEW/'outputs/wilor_clip2102/observations.json').read_text())['frames'];ids=[i for i,r in enumerate(rows) if r['frame_index']==0]
    surface=np.load(VIEW/'outputs/wilor_clip2102/surface.npz');rf=json.loads((VIEW/'outputs/rf_clip2102/predictions.json').read_text())['frames']
    with np.load(rf[0]['output']) as v:masks=v['masks']
    own_masks=masks&~(masks.sum(0)>1)[None];native=camera(frame['cameras']['214-1']['calibration']);centers=np.array([(np.array(rows[i]['box_xyxy'][:2])+rows[i]['box_xyxy'][2:])/2 for i in ids]);rays=np.stack([unproject(native,uv) for uv in centers])
    with tarfile.open(PUBLIC/'assets/clip-002102-full.tar') as tar:
        hands=json.load(tar.extractfile('000000.hands.json'));beta=json.load(tar.extractfile('__hand_shapes.json__'))['mano'];cal=json.load(tar.extractfile('000000.cameras.json'))['214-1'];pose=transform(cal['T_world_from_camera'])
    targets=[(side,np.array(h['boxes_amodal']['214-1'])) for side,h in hands.items() if '214-1' in h.get('boxes_amodal',{})]
    overlap=np.array([[iou(np.array(rows[i]['box_xyxy']),b) for side,b in targets] for i in ids]);a,b=linear_sum_assignment(1-overlap);right=[ids[x] for x,y in zip(a,b) if targets[y][0]=='right' and overlap[x,y]>=.1];assert len(right)==1;i=right[0]
    h=hands['right']['mano_pose'];root=torch.tensor([h['wrist_xform']],dtype=torch.float32);layer=smplx.create(str(common.RUN/'assets/MANO_RIGHT.pkl'),'mano',use_pca=True,is_rhand=True,num_pca_comps=15).eval()
    with torch.no_grad():gt=layer(betas=torch.tensor([beta],dtype=torch.float32),global_orient=root[:,:3],hand_pose=torch.tensor([h['thetas']],dtype=torch.float32),transl=root[:,3:],return_verts=True)
    gt_mesh=((gt.vertices[0].numpy()-pose[:3,3])@pose[:3,:3]).astype(np.float32);gt_wrist=(gt.joints[0,0].numpy()-pose[:3,3])@pose[:3,:3];results=[];all_pred=[];all_raw=[];all_offset=[]
    for target in ['1201-1','1201-2']:
        cache=VIEW/'outputs/depth_clip2102'/f'example_{target}.npz';v=np.load(cache);disparity,confidence=v['disparity'],v['confidence']
        spec=rectified_pair(frame['cameras'],'214-1',target);native_from_rect=spec['first_device_transform'][:3,:3].T@spec['device_from_rectified'];rect_rays=rays@native_from_rect;visible=rect_rays[:,2]>.05
        center=np.median(rect_rays[visible,:2]/rect_rays[visible,2:3],axis=0);principal=np.array([239.5,143.5])-spec['focal_px']*center
        maps=[];valid=[]
        for key in ['214-1',target]:
            extr=transform(frame['cameras'][key]['calibration']['T_device_from_camera']);uv,good=remap_geometry(frame['cameras'][key]['calibration'],480,288,spec['focal_px'],principal,extr[:3,:3].T@spec['device_from_rectified']);maps.append(uv);valid.append(good)
        yy,xx=np.mgrid[:288,:480];rx=np.rint(xx-disparity).astype(np.int32);supported=(rx>=0)&(rx<480);supported&=valid[1][yy,np.clip(rx,0,479)]
        good=valid[0]&supported&(confidence>=.2)&(disparity>.5)&np.isfinite(disparity);z=spec['focal_px']*spec['baseline_m']/np.maximum(disparity,1e-6)
        points=np.stack([(xx-principal[0])*z/spec['focal_px'],(yy-principal[1])*z/spec['focal_px'],z],-1)@native_from_rect.T
        px=np.rint(maps[0][...,0]).astype(np.int32);py=np.rint(maps[0][...,1]).astype(np.int32);good&=(px>=0)&(px<1408)&(py>=0)&(py<1408)&(points[...,2]>.08)&(points[...,2]<3.)
        assert np.array_equal(good,v['valid']),'Cached rectification/support contract changed'
        selected=good&own_masks[rows[i]['detection']][np.clip(py,0,1407),np.clip(px,0,1407)];loc=np.flatnonzero(selected);count=len(loc)
        if len(loc)<6:results.append(dict(target=target,candidate_pixels=count,skipped=True));continue
        loc=loc[np.linspace(0,len(loc)-1,min(96,len(loc)),dtype=int)];observed=points.reshape(-1,3)[loc].astype(np.float32);direction=observed/observed[:,2:3]
        predicted_hits=surface_intersections(direction,surface['vertices'][i],surface['faces']);truth_hits=surface_intersections(direction,gt_mesh,surface['faces']);offset=predicted_hits-surface['joints'][i,0,2];accepted=np.isfinite(predicted_hits)&(np.abs(offset)<.15)
        wrist=observed[:,2]-offset;accepted&=(wrist>.08)&(wrist<3.);both=accepted&np.isfinite(truth_hits)
        sensor_error=observed[both,2]-truth_hits[both];offset_error=(truth_hits[both]-gt_wrist[2])-offset[both];total=wrist[both]-gt_wrist[2]
        assert float(np.max(np.abs(total-sensor_error-offset_error),initial=0))<1e-6
        results.append(dict(target=target,candidate_pixels=count,sampled=len(loc),old_predicted_surface_hits=int(accepted.sum()),also_GT_mesh_hits=int(both.sum()),
            f_rect_px=float(spec['focal_px']),baseline_m=float(spec['baseline_m']),disparity=stats(disparity.reshape(-1)[loc][both]),
            stereo_surface_Z_signed_error_mm=stats(sensor_error*1000),surface_to_wrist_offset_signed_error_mm=stats(offset_error*1000),wrist_Z_signed_error_mm=stats(total*1000),
            predicted_wrist_median_m=float(np.median(wrist[accepted])) if accepted.any() else None,GT_wrist_Z_m=float(gt_wrist[2])))
        all_pred.extend(wrist[accepted]);all_raw.extend(sensor_error);all_offset.extend(offset_error)
    old=json.loads((VIEW/'outputs/depth_clip2102/depth.json').read_text())['observations'];anchor=next(x for x in old if x['observation']==i)
    aggregate=float(np.median(all_pred)) if all_pred else None
    if anchor['valid']:assert abs(aggregate-anchor['wrist_depth_m'])<1e-6
    result=dict(complete=True,posthoc_diagnostic_only=True,frame=0,hand_side='right only',observation=i,GT_wrist_camera=gt_wrist.tolist(),source_anchor=anchor,reproduced_anchor_median_m=aggregate,results=results,
        diagnostic_scope='Existing WiLoR RIGHT asset and original GT MANO parameters. GT surface is not a new inference input. Occlusion/outside-mesh rays reported as uncovered; not a claim full physical surface ground truth or independent original asset-version verification.',
        next='Only two cached stereo views at frame0; extend diagnosed cases before changing an engine or correction rule',production_admitted=False,
        run_depth_source_sha256=hashlib.sha256((PUBLIC/'src/run_depth.py').read_bytes()).hexdigest())
    (R/'surface_frame0.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
