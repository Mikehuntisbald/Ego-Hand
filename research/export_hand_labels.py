"""Export RGB-only clips to YOLO detection/20-point pose and 3D hand GT.

Uses the pinned official toolkit for UmeTrack FK and FISHEYE624 projection.
The modeled hand visibility is NOT a per-landmark occlusion annotation.
"""
import argparse, hashlib, io, json, os, re, tarfile, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from PIL import Image, ImageDraw
from hand_tracking_toolkit.dataset import decode_hand_pose
from hand_tracking_toolkit.camera import from_json as camera_from_json
from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json as model_from_json, forward_kinematics

ROOT = Path('/mnt/why/HOT3D')
NAMES = ['thumb_tip','index_tip','middle_tip','ring_tip','little_tip','wrist',
         'thumb_intermediate','thumb_distal','index_proximal','index_intermediate','index_distal',
         'middle_proximal','middle_intermediate','middle_distal','ring_proximal','ring_intermediate',
         'ring_distal','little_proximal','little_intermediate','little_distal']
EDGES = [[5,6],[6,7],[7,0],[5,8],[8,9],[9,10],[10,1],[5,11],[11,12],[12,13],[13,2],
         [5,14],[14,15],[15,16],[16,3],[5,17],[17,18],[18,19],[19,4]]

def write_json(path, value):
    part=path.with_suffix(path.suffix+'.partial')
    part.write_text(json.dumps(value,indent=2));part.replace(path)

def export_clip(job):
    torch.set_num_threads(1)
    split, subject, sequence, clip=job
    source=ROOT/'rgb_clips'/split/sequence/f'clip-{clip:06d}.tar'
    base=ROOT/'export';identifier=f'clip-{clip:06d}'
    image_dir=base/'images'/split/sequence/identifier
    detect_dir=base/'detect/labels'/split/sequence/identifier
    pose_dir=base/'pose/labels'/split/sequence/identifier
    annotation_dir=base/'annotations'/split/sequence;annotation_dir.mkdir(parents=True,exist_ok=True)
    annotation_file=annotation_dir/f'{identifier}.jsonl'
    receipt_file=annotation_dir/f'{identifier}.receipt.json'
    input_receipt=json.loads(source.with_suffix('.receipt.json').read_text())
    if receipt_file.exists():
        result=json.loads(receipt_file.read_text())
        assert result['source_retained_sha256']==input_receipt['retained_archive_sha256']
        assert result['frames']==150 and annotation_file.exists()
        assert len(list(image_dir.glob('*.jpg')))==150
        assert len(list(detect_dir.glob('*.txt')))==150
        assert len(list(pose_dir.glob('*.txt')))==150
        return result
    for p in [image_dir,detect_dir,pose_dir]:p.mkdir(parents=True,exist_ok=True)
    frames=hands_count=label_count=keypoints_count=valid_count=inside_box=0
    roundtrip_error=0.;depths=[];timestamps=[];visibilities=[];unlabeled_frames=0
    out_part=annotation_file.with_suffix('.jsonl.partial')
    with tarfile.open(source) as archive, out_part.open('w') as output:
        entries={m.name:m for m in archive.getmembers()}
        assert len(entries)==601 and all(n.endswith(('.image_214-1.jpg','.cameras.json','.hands.json','.info.json')) or n=='__hand_shapes.json__' for n in entries)
        def load(name):return json.loads(archive.extractfile(entries[name]).read())
        shapes=load('__hand_shapes.json__');model=model_from_json(shapes['umetrack'])
        for frame in range(150):
            stem=f'{frame:06d}';meta=load(stem+'.info.json');raw_cam=load(stem+'.cameras.json')['214-1']
            assert meta['participant_id']==subject and meta['sequence_id']==sequence and meta['device']=='Aria'
            timestamp=meta['image_timestamps_ns']['214-1'];timestamps.append(timestamp)
            camera=camera_from_json(raw_cam);w,h=camera.width,camera.height
            assert (w,h)==(1408,1408)
            image_data=archive.extractfile(entries[stem+'.image_214-1.jpg']).read()
            img_path=image_dir/(stem+'.jpg')
            # Verify every JPEG before exposing it as a completed exported clip.
            img=Image.open(io.BytesIO(image_data));assert img.size==(w,h);img.load()
            img_path.write_bytes(image_data)
            raw_hands=load(stem+'.hands.json');poses=decode_hand_pose(raw_hands)
            detect_lines=[];pose_lines=[];hand_records=[]
            for side,collection in poses.items():
                if collection.umetrack is None:continue
                world=forward_kinematics(collection.umetrack,model,requires_mesh=False)[0].detach().numpy()
                assert world.shape==(20,3) and np.isfinite(world).all()
                xyz=camera.world_to_eye(world)
                uv=camera.eye_to_window(xyz)
                assert np.isfinite(xyz).all() and np.isfinite(uv).all()
                roundtrip_error=max(roundtrip_error,float(np.max(np.abs(camera.eye_to_world(xyz)-world))))
                angle=np.arctan2(np.linalg.norm(xyz[:,:2],axis=1),xyz[:,2])
                valid=(xyz[:,2]>0)&(angle<=raw_cam['calibration']['max_solid_angle'])&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
                gt=raw_hands[side.value];fraction=gt.get('visibilities_modeled',{}).get('214-1')
                box=gt.get('boxes_amodal',{}).get('214-1')
                record=dict(side=side.value,xyz_world_m=world.tolist(),xyz_camera_m=xyz.tolist(),
                            xyz_wrist_relative_m=(xyz-xyz[5]).tolist(),uv_pixels=uv.tolist(),
                            keypoint_projection_valid=valid.tolist(),keypoint_occlusion_visibility=None,
                            modeled_hand_visible_fraction=fraction,box_amodal_xyxy=box,
                            umetrack_pose=gt['umetrack_pose'],mano_pose=gt.get('mano_pose'))
                hand_records.append(record);hands_count+=1;keypoints_count+=20;valid_count+=int(valid.sum())
                if fraction is not None:visibilities.append(float(fraction))
                depths.extend(xyz[:,2].tolist())
                if box is None:continue
                clipped=np.clip(np.asarray(box,dtype=float),[0,0,0,0],[w,h,w,h])
                x1,y1,x2,y2=clipped
                # Amodal box and pose labels for observable hands only. The GT of
                # fully hidden/out-of-frame hands is retained in the 3D records.
                if x2-x1<2 or y2-y1<2 or (fraction is not None and fraction<=0):continue
                center=[(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h]
                bbox='0 '+' '.join(f'{v:.8f}' for v in center)
                detect_lines.append(bbox);label_count+=1
                kp=[]
                for p,v in zip(uv,valid):
                    # v=1 means labeled, with visibility unknown; v=2 would assert
                    # observed visibility that HOT3D does not annotate per point.
                    kp.extend([p[0]/w,p[1]/h,1] if v else [0,0,0])
                pose_lines.append(bbox+' '+' '.join(f'{v:.8f}' for v in kp))
                in_box=(uv[:,0]>=x1-5)&(uv[:,0]<=x2+5)&(uv[:,1]>=y1-5)&(uv[:,1]<=y2+5)
                inside_box+=int((in_box&valid).sum())
            if not detect_lines:unlabeled_frames+=1
            (detect_dir/(stem+'.txt')).write_text('\n'.join(detect_lines)+'\n' if detect_lines else '')
            (pose_dir/(stem+'.txt')).write_text('\n'.join(pose_lines)+'\n' if pose_lines else '')
            record=dict(split=split,subject=subject,sequence=sequence,clip=clip,frame=frame,
                        timestamp_ns=timestamp,image=str(img_path.relative_to(base)),
                        image_size=[w,h],camera=raw_cam,hand_model_profile_ref=str(source)+':__hand_shapes.json__',
                        hands=hand_records)
            output.write(json.dumps(record,separators=(',',':'))+'\n');frames+=1
            if frame in (0,75) and clip in (1892,):
                draw=ImageDraw.Draw(img)
                for hand in hand_records:
                    color='#22dd88' if hand['side']=='left' else '#ff9955'
                    if hand['box_amodal_xyxy']:draw.rectangle(hand['box_amodal_xyxy'],outline=color,width=4)
                    points=np.asarray(hand['uv_pixels']);valid=np.asarray(hand['keypoint_projection_valid'])
                    for a,b in EDGES:
                        if valid[a] and valid[b]:draw.line([tuple(points[a]),tuple(points[b])],fill=color,width=4)
                    for i,((x,y),v) in enumerate(zip(points,valid)):
                        if v:draw.ellipse((x-4,y-4,x+4,y+4),fill=color);draw.text((x+6,y),str(i),fill=color)
                previews=ROOT/'previews';previews.mkdir(exist_ok=True)
                img.resize((704,704)).save(previews/f'{identifier}-{frame:06d}-gt.jpg',quality=90)
    assert frames==150 and all(b>a for a,b in zip(timestamps,timestamps[1:]))
    assert roundtrip_error<1e-5
    out_part.replace(annotation_file)
    result=dict(completed=True,source_retained_sha256=input_receipt['retained_archive_sha256'],
                split=split,subject=subject,sequence=sequence,clip=clip,frames=frames,hands=hands_count,
                detector_boxes=label_count,keypoints=keypoints_count,valid_projected_keypoints=valid_count,
                keypoints_inside_labeled_boxes_with_5px_margin=inside_box,
                frames_without_detector_labels=unlabeled_frames,
                xyz_roundtrip_max_error_m=roundtrip_error,
                camera_depth_m_quantiles=np.quantile(depths,[0,.05,.5,.95,1]).tolist() if depths else [],
                hand_visibility_quantiles=np.quantile(visibilities,[0,.25,.5,.75,1]).tolist() if visibilities else [],
                annotation_sha256=hashlib.sha256(annotation_file.read_bytes()).hexdigest())
    write_json(receipt_file,result);return result

def setup():
    base=ROOT/'export';base.mkdir(exist_ok=True)
    for task in ['detect','pose']:
        p=base/task;p.mkdir(exist_ok=True)
        link=p/'images'
        if not link.exists() and not link.is_symlink():link.symlink_to('../images',target_is_directory=True)
        data=dict(path=str(p),train='train.txt',val='val.txt',names={0:'hand'})
        if task=='pose':data.update(kpt_shape=[20,3],flip_idx=list(range(20)),kpt_names={0:NAMES})
        import yaml
        (p/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    write_json(base/'keypoint_schema.json',dict(names=NAMES,edges=EDGES,wrist_index=5,coordinate_units='meters',
        xyz_camera_axes='+X right, +Y down, +Z forward',projection='official FISHEYE624; original 1408x1408 RGB',
        visibility='0 invalid projection, 1 labeled with occlusion visibility unknown; never fabricate v=2',
        split='subject-disjoint validation held out from official training; official test GT unavailable',
        gt_use='labels and evaluation only; validation crops and coarse poses must come from predictions'))

def rebuild_lists(rows):
    base=ROOT/'export'
    for task in ['detect','pose']:
        for split in ['train','val']:
            files=[]
            for row in sorted(rows,key=lambda r:(r['sequence'],r['clip'])):
                if row['split']==split:
                    folder=base/task/'images'/split/row['sequence']/f"clip-{row['clip']:06d}"
                    files.extend(str(folder/f'{frame:06d}.jpg') for frame in range(150))
            dest=base/task/f'{split}.txt';part=dest.with_suffix('.txt.partial')
            part.write_text('\n'.join(files)+'\n' if files else '');part.replace(dest)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');ap.add_argument('--workers',type=int,default=4);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    setup();manifest=json.loads((ROOT/'subset_manifest.json').read_text())
    jobs=[(s['split'],s['subject'],s['sequence'],c) for s in manifest['sequences'] for c in s['clips']]
    done={};errors={};started=time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        while True:
            available=[j for j in jobs if j not in done and (ROOT/'rgb_clips'/j[0]/j[2]/f'clip-{j[3]:06d}.receipt.json').exists()]
            if a.limit:available=available[:a.limit]
            futures={pool.submit(export_clip,j):j for j in available}
            for future in as_completed(futures):
                job=futures[future]
                try:done[job]=future.result();errors.pop(str(job),None)
                except Exception as e:errors[str(job)]=repr(e)
                status=dict(stage='exporting',completed_clips=len(done),expected_clips=len(jobs),frames=sum(r['frames'] for r in done.values()),hands=sum(r['hands'] for r in done.values()),errors=errors,seconds=time.time()-started)
                write_json(ROOT/'export_status.json',status);print(json.dumps(status),flush=True)
            rows=list(done.values());rebuild_lists(rows)
            if a.limit or not a.watch or len(done)==len(jobs):break
            time.sleep(20)
    status=dict(stage='complete' if len(done)==len(jobs) else 'pilot_complete',completed_clips=len(done),expected_clips=len(jobs),
                frames=sum(r['frames'] for r in done.values()),hands=sum(r['hands'] for r in done.values()),
                train_frames=sum(r['frames'] for r in done.values() if r['split']=='train'),
                val_frames=sum(r['frames'] for r in done.values() if r['split']=='val'),errors=errors,seconds=time.time()-started)
    write_json(ROOT/'export_status.json',status);print(json.dumps(status),flush=True)
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
