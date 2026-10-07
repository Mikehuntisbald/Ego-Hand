"""Freeze a finite, paired RF conditioning adaptation and complete-pipeline replay.

The input manifests contain only observation metadata. Existing test footage is
diagnostic replay, never described as a new generalization test.
"""
import collections, hashlib, json, os, shutil, time
from pathlib import Path
import numpy as np
import torch

E=Path('/mnt/why/HOT3D/experiments')
ROOT=E/'rfdetr_fair_3d_v54r1_20261008'
OLD=E/'full_model_gloves_multihand_v51_20261007'
SOURCE=E/'online_rgb_3d_v47'
CODE=Path('/mnt/why/hot3d_hand_residual')
RF=E/'rfdetr_multidata_v53B_20261007/strict_review_r1/best_development_ema.pth'
INITIAL=OLD/'paired_protocol/core_r1/dit_joint/best.pt'
def save(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2),encoding='utf-8');temp.replace(path)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def link(src,dst):
    dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists():os.symlink(src,dst)
def main():
    torch.set_num_threads(4);ROOT.mkdir(exist_ok=True)
    if (ROOT/'prepared.json').exists():print((ROOT/'prepared.json').read_text());return
    snapshot=ROOT/'code';snapshot.mkdir(exist_ok=True)
    for src in CODE.glob('*.py'):shutil.copy2(src,snapshot/src.name)
    masks=torch.load(OLD/'mask_conditions.pt',weights_only=False,mmap=True)
    data=torch.load(SOURCE/'inputs.pt',weights_only=False,mmap=True)
    records=json.loads((SOURCE/'records.json').read_text())
    indices=masks['indices'];fids=torch.unique(data['feature_ids'][indices['train']+indices['dev_select']]).tolist();fids=[x for x in fids if x]
    grouped=collections.defaultdict(list)
    for fid in fids:grouped[records[fid-1]['image']].append(fid)
    frames=[]
    for i,(image,ids) in enumerate(sorted(grouped.items())):
        r=records[ids[0]-1]
        frames.append(dict(id=f'native_{i}',image=image,camera=r['camera'],timestamp_s=0.,camera_calibrated=True))
    save(ROOT/'native_adaptation_rgb.json',dict(frames=frames))
    save(ROOT/'native_feature_map.json',dict(fids=fids,images={r['image']:grouped[r['image']] for r in frames},indices=indices))
    # Existing normalized training/dev RGB, in the exact original cache order.
    raw=[json.loads(x) for x in (E.parent/'domain_data_v51/surgical_hands/selected_records.jsonl').read_text().splitlines()]
    raw=[r for r in raw if r['split']!='test'];external=[];mapping=[];counter=0
    from annotate_instances_v51_r6 import normalize_frame
    normal=ROOT/'surgical_normalized';normal.mkdir(exist_ok=True)
    for i,row in enumerate(raw):
        f=normalize_frame(dict(image=row['image'],timestamp_s=0.),normal,i)
        f.update(id=row['id'],group=row['group'],split=row['split'])
        external.append(f)
        for hand in row['hands']:
            mapping.append(dict(index=counter,image=f['image'],id=hand['id'],group=row['group'],split=row['split'],
                                human_box=(np.asarray(hand['bbox'])*f['image_transform']['scale']+np.tile(f['image_transform']['offset'],2)).tolist()))
            counter+=1
    old_meta=torch.load(OLD/'surgical_pose/inputs.pt',weights_only=False,mmap=True)['metadata']
    assert len(mapping)==len(old_meta) and all(a['id']==b['id'] and a['split']==b['split'] for a,b in zip(mapping,old_meta))
    save(ROOT/'surgical_adaptation_rgb.json',dict(frames=external))
    save(ROOT/'surgical_supervision_map.json',mapping)
    # No GT, handedness, GT identity or annotated hand boxes enter replay input.
    replay=json.loads((SOURCE/'diagnostic_v46/rows.json').read_text())
    groups=collections.defaultdict(dict)
    for r in replay:groups[(r['sequence'],r['clip'])][r['frame']]=r
    clips=[]
    for (seq,clip),items in sorted(groups.items()):
        fs=[dict(id=f'{seq}_{clip}_{r["frame"]}',image=r['image'],camera=r['camera'],camera_calibrated=True,
                 timestamp_s=r['timestamp_ns']/1e9) for _,r in sorted(items.items())]
        tag=f'{seq}_{clip}';save(ROOT/'replay'/f'{tag}.json',dict(frames=fs));clips.append(dict(id=tag,sequence=seq,clip=clip,input=str(ROOT/'replay'/f'{tag}.json'),frames=len(fs)))
    glove=json.loads((E/'rfdetr_multidata_v53B_20261007/glove_rgb.json').read_text())
    save(ROOT/'glove_replay_rgb.json',glove);save(ROOT/'replay_clips.json',clips)
    nail=json.loads((E/'rfdetr_multidata_v53B_20261007/nail_rgb.json').read_text());save(ROOT/'replay/nail.json',nail)
    for arm in ['yolo_condition_control','rf_condition_adapt']:
        root=ROOT/arm
        for name in ['domain_masks.pt','domain_pixels.npy']:link(OLD/'paired_protocol'/name,root/name)
        link(OLD/'surgical_pose/targets.pt',root/'surgical_pose/targets.pt')
        for name in ['targets.pt','risk.pt','pixels.npy']:link(SOURCE/name,root/'native'/name)
    # Exact same native/surgical/Ego sample plan, seed, loss and budget.
    protocol=dict(version=54,created_unix=time.time(),finite=True,updates_per_arm=160,preflight_updates=20,seed=2026100751,
      initial=str(INITIAL),initial_sha256=sha(INITIAL),RF_checkpoint=str(RF),RF_sha256=sha(RF),RF_threshold=.2,RF_box_nms=.7,
      matched_training=['identical_initial_weights','same_window_and_external_instance_ids','same_seed_and_batch_plan','same_losses_and_learning_rates','same_update_count'],
      adaptation='Live RGB final four blocks + instance-conditioned temporal DiT/FK. RF boxes replace visual crop/positions/mask conditions; common original WiLoR/IK training observations stay fixed, as in v47. External single-frame RGB crops also use matched RF predictions.',
      external_matching='Annotated boxes only assign training/dev supervision to already-frozen RF predictions. No annotated ROI is passed to RF inference. Both arms use the same matched external instances.',
      coarse_training_observations_fixed=True,external_3D_GT=False,external_timestamps_fabricated=False,
      supervision='Native real 3D/diffusion/teacher-good-point protection + genuine surgical 2D/visibility/side loss + human Ego mask loss. GT/teacher loss-only.',
      context_s=1.6,noncausal=True,spatial_tokens=192,speed_limits_unchanged=True,soft_hard_acceleration_x2=True,synthetic_occlusion=False,
      inference_modes=['yolo_v43','yolo_v48','rf_v48','rf_v53','rf_v54','rf_v54_side_locked','rf_v54_best','yolo_v54_control'],
      common_runtime='Same images/timestamps/cameras/GPU3/BF16, 4 draws, 10 DDIM steps, same IK and speed/acceleration constraints. v43 weights evaluated with the common +/-1.6s, acceleration-x2 runtime.',
      association='Same box-motion/box-IoU/WiLoR appearance association for all frontends; masks enter the reconstruction conditions, not association in this controlled replay.',
      frontend_comparison='Historical HOT3D YOLO checkpoint vs fixed v53 RF. Training control uses original common YOLO crop observations. No claim of equal total historical detector training data/budget.',
      side_locked='Diagnostic output chirality locked to the predicted track-consensus coarse side; does not correct a wrong initial side or use GT.',
      evaluation=['all-GT-hand PCK20 with misses counted wrong','common-matched-point camera/relative MPJPE','glove all-point projection PCK10','old-correct-point damage and hard-point recovery','source-sequence paired 95% bootstrap CI','ID fragmentation and switches where native labels permit','jumps and bone consistency','fast-motion amplitude retention'],
      native_GT_coverage='All finite hand landmarks in original export annotations. Association uses tight projected GT boxes; amodal projection coverage is not visible-pixel detector recall.',
      output_freeze_before_test_labels=True,training_selection='Original <=1% WiLoR/teacher good-point harm gates retained; failed training candidates are diagnostic, never safe best.',
      matched_training_endpoint='Fixed last160 checkpoints for both training arms; separately evaluate admitted best. Fixed-last and side-lock trials remain diagnostic until all gates pass.',
      replay_scope='Existing 10 HOT3D clips / five sequences / two known subjects and 12 surgical stills / six source groups. Reused diagnostic replay; no fresh generalization claim.',
      qualitative_tags=['nail'],nail_scope='Existing real 120-frame nail-care replay, no GT person IDs or 3D pose. Only numeric constraints and qualitative masks/pose/fragmentation.',
      native_clips=len(clips),native_frames=sum(x['frames'] for x in clips),glove_images=len(glove['frames']),native_train_windows=len(indices['train']),native_dev_windows=len(indices['dev_select']),
      default_changed=False,automation_restarted=False)
    save(ROOT/'protocol.json',protocol)
    hashes={p.name:sha(p) for p in snapshot.glob('*.py')};save(ROOT/'code_hashes.json',hashes)
    save(ROOT/'prepared.json',dict(complete=True,protocol_sha256=sha(ROOT/'protocol.json'),native_images=len(frames),external_images=len(external),external_instances=len(mapping),snapshot_files=len(hashes)))
    print(json.dumps(protocol),flush=True)
if __name__=='__main__':main()
