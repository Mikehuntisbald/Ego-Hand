"""Training manifest/GT warmup crops; no held-out GT boxes enter evaluation."""
import hashlib,json,random,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import cv2,numpy as np
from hand_tracking_toolkit.camera import from_json
ROOT=Path('/mnt/why/HOT3D');RUN=ROOT/'experiments/dit_lowconfidence_v1'

def unproject(camera,uv):
    # Invert only the distortion in the official camera implementation.
    target=(np.asarray(uv)-camera.c)/camera.f
    direction=target/max(np.linalg.norm(target),1e-9)
    p=direction*min(np.linalg.norm(target),.65)
    # A partially observed hand may have its amodal box center outside the
    # calibrated sensor cone. Use the nearest valid ray for root initialization;
    # the network still receives the original ROI coordinates and learns offsets.
    limit=1.0
    for _ in range(20):
        fx=camera.distort.evaluate(p)-target
        if np.max(np.abs(fx))<1e-9:break
        eps=1e-5
        j=np.stack([(camera.distort.evaluate(p+np.array([eps,0]))-camera.distort.evaluate(p-np.array([eps,0])))/(2*eps),
                    (camera.distort.evaluate(p+np.array([0,eps]))-camera.distort.evaluate(p-np.array([0,eps])))/(2*eps)],axis=-1)
        step=np.linalg.solve(j,fx);step*=min(1.,.15/max(np.linalg.norm(step),1e-9))
        p-=step;p*=min(1.,limit/max(np.linalg.norm(p),1e-9))
    ray=camera.unproject(p)
    assert np.isfinite(ray).all()
    projected=camera.eye_to_window(ray)
    if np.linalg.norm(p)<limit-.005:assert np.linalg.norm(projected-uv)<.01,(projected,uv)
    return ray

def crop_and_geometry(image,box,camera_json,size=256):
    cam=from_json(camera_json);box=np.asarray(box,dtype=float)
    center=(box[:2]+box[2:])/2;side=max(box[2]-box[0],box[3]-box[1])*1.3
    side=max(side,24);roi=np.r_[center-side/2,center+side/2]
    scale=size/side
    # Pixel centers: this matches the continuous ROI coordinates in supervision.
    affine=np.array([[scale,0,(.5-roi[0])*scale-.5],[0,scale,(.5-roi[1])*scale-.5]],dtype=np.float32)
    crop=cv2.warpAffine(image,affine,(size,size),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
    params=np.array(camera_json['calibration']['projection_params'],dtype=float)
    intrinsics=params.copy();intrinsics[0]/=1408;intrinsics[1:3]/=1408
    ray=unproject(cam,center)
    geometry=np.r_[center/1408,side/1408,intrinsics,ray].astype(np.float32)
    assert geometry.shape==(21,)
    return crop,roi.astype(np.float32),geometry

def process_clip(args):
    path,role=args;path=Path(path);rows=[];frames=path.read_text().splitlines()
    for frame,line in enumerate(frames):
        # GT crops only warm up coarse training, never tune/test/inference.
        row=json.loads(line)
        stride=5 if role=='coarse' else 3
        if role!='coarse' or frame%stride:continue
        image=cv2.imread(str(ROOT/'export'/row['image']));assert image is not None
        cam=row['camera']
        for hand in row['hands']:
            box=hand['box_amodal_xyxy'];vis=hand['modeled_hand_visible_fraction'];xyz=np.array(hand['xyz_camera_m'],np.float32)
            if box is None or vis is None or vis<=0 or xyz[5,2]<=.05:continue
            box=np.clip(box,[0,0,0,0],[1408,1408,1408,1408])
            if min(box[2]-box[0],box[3]-box[1])<2:continue
            crop,roi,geometry=crop_and_geometry(image,box,cam)
            folder=RUN/'warmup_crops'/row['sequence']/f"clip-{row['clip']:06d}";folder.mkdir(parents=True,exist_ok=True)
            name=f"{frame:06d}_{hand['side']}.jpg";dest=folder/name
            assert cv2.imwrite(str(dest),crop,[cv2.IMWRITE_JPEG_QUALITY,95])
            record=dict(role=role,subject=row['subject'],sequence=row['sequence'],clip=row['clip'],frame=frame,
                side=hand['side'],crop=str(dest),image=str(ROOT/'export'/row['image']),roi=roi.tolist(),geometry=geometry.tolist(),
                xyz_camera_m=xyz.tolist(),uv_pixels=hand['uv_pixels'],projection_valid=hand['keypoint_projection_valid'],
                visible_fraction=vis,box_score=1.,box_source='GT_warmup_training_only',timestamp_ns=row['timestamp_ns'])
            rows.append(record)
    return rows

def main():
    RUN.mkdir(parents=True,exist_ok=True);m=json.loads((ROOT/'subset_manifest.json').read_text());rng=random.Random(20261003)
    coarse=[];residual=[];tune=[];test=[]
    for s in m['sequences']:
        if s['split']=='train':
            ids=s['clips'].copy();rng.shuffle(ids);held=set(ids[:max(2,round(.2*len(ids)))])
            coarse.extend(c for c in s['clips'] if c not in held);residual.extend(held)
        elif s['subject']=='P0003':tune.extend(s['clips'])
        else:test.extend(s['clips'])
    assert not set(coarse)&set(residual) and not (set(coarse)|set(residual))&(set(tune)|set(test))
    partition=dict(seed=20261003,coarse_clips=sorted(coarse),residual_clips=sorted(residual),tune_clips=sorted(tune),
        locked_test_clips=sorted(test),tune_subjects=['P0003'],locked_test_subjects=['P0010','P0015'],
        note='Common coarse uses C clips; every learned refiner/control uses the same disjoint R clips. Final inference uses predicted boxes.')
    (RUN/'partition.json').write_text(json.dumps(partition,indent=2))
    jobs=[]
    for s in m['sequences']:
        for c in s['clips']:
            if c in coarse:jobs.append((str(ROOT/'export/annotations'/s['split']/s['sequence']/f'clip-{c:06d}.jsonl'),'coarse'))
    all_rows=[];started=time.time()
    with ProcessPoolExecutor(max_workers=12) as pool:
        fs=[pool.submit(process_clip,j) for j in jobs]
        for i,f in enumerate(as_completed(fs)):
            all_rows.extend(f.result())
            if (i+1)%30==0:print(json.dumps(dict(clips=i+1,samples=len(all_rows),seconds=time.time()-started)),flush=True)
    all_rows.sort(key=lambda r:(r['sequence'],r['clip'],r['frame'],r['side']))
    out=RUN/'warmup_manifest.jsonl';out.write_text('\n'.join(json.dumps(r,separators=(',',':')) for r in all_rows)+'\n')
    summary=dict(completed=True,samples=len(all_rows),coarse_clips=len(coarse),residual_clips=len(residual),
        tune_clips=len(tune),test_clips=len(test),manifest_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    (RUN/'prepare_pose_done.json').write_text(json.dumps(summary,indent=2));print(summary,flush=True)

if __name__=='__main__':main()
