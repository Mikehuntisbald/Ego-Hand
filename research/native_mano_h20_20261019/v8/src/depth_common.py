"""GT-free reconstruction of existing calibrated stereo points and mesh hits."""
import sys
from pathlib import Path
import cv2,numpy as np
PUBLIC=Path('/mnt/why/public_headhand_h20_20261019');VIEW=Path('/mnt/why/native_mano_dit_v2_20261019/coherent_frontend')
sys.path.insert(0,str(PUBLIC/'src'));sys.path.insert(0,str(PUBLIC/'vendor/Ego-Hand/research'))
from geometry import camera,rectified_pair,remap_geometry,transform
from wilor_eval_common import unproject

def hits(directions,vertices,faces):
    triangles=vertices[faces];v0=triangles[:,0];e1,e2=triangles[:,1]-v0,triangles[:,2]-v0
    p=np.cross(directions[:,None],e2[None]);det=(p*e1[None]).sum(-1)
    inv=np.divide(1.,det,out=np.zeros_like(det),where=np.abs(det)>1e-9)
    u=(p*(-v0)[None]).sum(-1)*inv;q=np.cross(-v0,e1);v=(directions[:,None]*q[None]).sum(-1)*inv
    distance=(e2*q).sum(-1)[None]*inv;valid=(np.abs(det)>1e-9)&(u>=0)&(v>=0)&(u+v<=1)&(distance>.01)
    return np.where(valid,distance,np.inf).min(-1)

def build_views(frame,ids,rows,cache_root):
    native=camera(frame['cameras']['214-1']['calibration'])
    centers=np.array([(np.array(rows[i]['box_xyxy'][:2])+rows[i]['box_xyxy'][2:])/2 for i in ids])
    rays=np.stack([unproject(native,uv) for uv in centers]);views=[]
    for target in ['1201-1','1201-2']:
        with np.load(cache_root/f"frame{frame['frame_index']:03d}_{target}.npz") as v:disparity=v['disparity'].copy();confidence=v['confidence'].copy();saved_valid=v['valid'].copy()
        spec=rectified_pair(frame['cameras'],'214-1',target);rotation=spec['first_device_transform'][:3,:3].T@spec['device_from_rectified']
        ray_rect=rays@rotation;visible=ray_rect[:,2]>.05;assert visible.any()
        center=np.median(ray_rect[visible,:2]/ray_rect[visible,2:3],axis=0);principal=np.array([239.5,143.5])-spec['focal_px']*center
        maps=[];valid=[]
        for key in ['214-1',target]:
            extr=transform(frame['cameras'][key]['calibration']['T_device_from_camera']);uv,good=remap_geometry(frame['cameras'][key]['calibration'],480,288,spec['focal_px'],principal,extr[:3,:3].T@spec['device_from_rectified']);maps.append(uv);valid.append(good)
        yy,xx=np.mgrid[:288,:480];rx=np.rint(xx-disparity).astype(np.int32);support=(rx>=0)&(rx<480);support&=valid[1][yy,np.clip(rx,0,479)]
        good=valid[0]&support&(confidence>=.2)&(disparity>.5)&np.isfinite(disparity);z=spec['focal_px']*spec['baseline_m']/np.maximum(disparity,1e-6)
        points=np.stack([(xx-principal[0])*z/spec['focal_px'],(yy-principal[1])*z/spec['focal_px'],z],-1)@rotation.T
        px=np.rint(maps[0][...,0]).astype(np.int32);py=np.rint(maps[0][...,1]).astype(np.int32)
        good&=(px>=0)&(px<1408)&(py>=0)&(py<1408)&(points[...,2]>.08)&(points[...,2]<3.)
        assert np.array_equal(good,saved_valid),'stereo validity parity failed'
        views.append(dict(target=target,points=points,good=good,px=px,py=py,confidence=confidence,disparity=disparity,spec=spec))
    return views

def select(view,mask):
    good=view['good']&mask[np.clip(view['py'],0,1407),np.clip(view['px'],0,1407)];loc=np.flatnonzero(good);count=len(loc)
    if count<6:return np.empty((0,3),np.float32),loc,count
    loc=loc[np.linspace(0,count-1,min(96,count),dtype=int)]
    return view['points'].reshape(-1,3)[loc].astype(np.float32),loc,count

def anchor(points,vertices,faces,wrist_z):
    if not len(points):return dict(valid=False,depth_m=None,samples=0,MAD_m=None),np.array([],bool),np.array([])
    direction=points/points[:,2:3];surface=hits(direction,vertices,faces);offset=surface-wrist_z;accepted=np.isfinite(surface)&(np.abs(offset)<.15)
    values=points[:,2]-offset;accepted&=(values>.08)&(values<3.)
    kept=values[accepted];median=float(np.median(kept)) if len(kept) else None;mad=float(np.median(np.abs(kept-median))) if len(kept) else None
    valid=len(kept)>=6 and mad<.03
    return dict(valid=bool(valid),depth_m=median if valid else None,samples=len(kept),MAD_m=mad),accepted,surface
