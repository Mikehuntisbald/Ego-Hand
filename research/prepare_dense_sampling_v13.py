"""Real30FPS observation preparation on existing training/development clips.

Reuse the audited prediction-only detector/box tracker/XYZ/RGB worker. No new
pixels are masked and no annotation field participates in box association.
Centers stay every5frames; near context becomes real1/30s with sparse far slots.
"""
import json,time,hashlib,collections
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,load,save
from hand3d_data_v7 import camera_pose
import spatial_rgb_common as s
ROOT=s.common.ROOT;RUN=V7.parent/'dense_sampling_v13'
OFFSETS=[-120,-60,-30,-10,-5,-3,-2,-1,0,1,2,3,5,10,30,60,120]

def freeze():
    RUN.mkdir(exist_ok=True);records,_=s.records_and_index();old=load('cpu');clips={}
    for i,role in enumerate(old['roles']):
        if role not in ['train','dev_select','dev_calibrate']:continue
        r=records[int(old['feature_ids'][i,8])-1];key=(r['sequence'],int(r['clip']));split=Path(r['image']).relative_to(ROOT/'export/images').parts[0]
        if key in clips:assert clips[key]['role']==role
        else:clips[key]=dict(sequence=key[0],clip=key[1],subject=old['subjects'][i],split=split,role=role)
    entries=[];grouped=collections.defaultdict(list)
    for clip in clips.values():
        path=ROOT/'export/annotations'/clip['split']/clip['sequence']/f"clip-{clip['clip']:06d}.jsonl"
        assert path.exists();lines=path.read_text().splitlines();assert len(lines)==150
        grouped[(clip['split'],clip['subject'],clip['sequence'])].append(clip['clip'])
    for (split,subject,sequence),ids in sorted(grouped.items()):entries.append(dict(split=split,subject=subject,sequence=sequence,clips=sorted(ids)))
    common=json.loads((ROOT/'subset_manifest.json').read_text());manifest={k:common[k] for k in ['official_repo','revision','mirror','stream']};manifest.update(sequences=entries,created_unix=time.time(),frames_stride=1,center_stride=5,context_frame_offsets=OFFSETS,scope='Existing86training/development clips only. This is denser observation preparation, not an independent new benchmark.',natural_only=True)
    path=RUN/'fresh_manifest.json'
    if path.exists():assert json.loads(path.read_text())['sequences']==entries
    else:save(path,manifest)
    save(RUN/'source_roles.json',list(clips.values()));save(RUN/'fresh_download_done.json',dict(complete=True,source_exports_reused=True,clips=len(clips),frames=150*len(clips),network_download=False))
    return clips

def prepare_observations():
    worker=Path(__file__).resolve().parent/'prepare_fresh_hand3d_v8.py';original=worker.read_text();modified=original
    replacements={
        'if i%5==0':'if True',
        "if not r['matched'] or uv_valid_bank[center].sum()<10:continue":"if not r['matched'] or uv_valid_bank[center].sum()<10 or r['frame']%5!=0:continue",
        "for offset in OFFSETS['multiscale']:":f'for frame_offset in {OFFSETS}:',
        "fid=group.get(r['frame']+5*offset,0)":"fid=group.get(r['frame']+frame_offset,0)",
        'else offset/6)':'else frame_offset/30)'}
    for before,after in replacements.items():assert modified.count(before)==1,(before,modified.count(before));modified=modified.replace(before,after)
    generated=RUN/'observation_worker_v13_snapshot.py';generated.write_text(modified)
    save(RUN/'worker_provenance.json',dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),generated_sha256=hashlib.sha256(modified.encode()).hexdigest(),replacements=replacements,natural_only=True))
    namespace=dict(__name__='dense_sampling_v13_worker',__file__=str(generated));exec(compile(modified,str(generated),'exec'),namespace);namespace['RUN']=RUN;namespace['prepare']()

def assemble_roles_and_targets(clips):
    data=torch.load(RUN/'fresh_data.pt',weights_only=False,mmap=True);records=json.loads((RUN/'fresh_rows.json').read_text());n=len(records)+1
    data['roles']=[clips[(r['sequence'],int(r['clip']))]['role'] for r in data['rows']]
    for row in data['rows']:row['source']='dense_sampling_v13'
    gt=torch.zeros(n,20,3);valid=torch.zeros(n,20,dtype=torch.bool);uv=torch.zeros(n,20,2);uv_valid=torch.zeros(n,20,dtype=torch.bool)
    for fid,r in enumerate(records,1):
        if r['matched']:
            gt[fid]=torch.tensor(r['gt']);valid[fid]=torch.isfinite(gt[fid]).all(-1);xy=s.common.from_json(r['camera']).eye_to_window(np.asarray(r['gt']))/1408;uv[fid]=torch.tensor(np.nan_to_num(xy));uv_valid[fid]=torch.tensor(r['projection_valid'])&torch.tensor(np.isfinite(xy).all(-1))
    world=torch.einsum('njc,nkc->njk',gt,data['rotation'])+data['translation'][:,None]
    extra=dict(gt_world=world,valid=valid,gt_uv=uv,uv_valid=uv_valid);center=data['feature_ids'][:,8];reconstructed=torch.einsum('njc,nck->njk',world[center]-data['translation'][center,None],data['rotation'][center]);error=float((reconstructed-data['gt']).abs().max());assert error<2e-6
    torch.save(data,RUN/'data.pt');torch.save(extra,RUN/'trajectory_targets.pt')
    observed=data['feature_ids']>0;near=[7,9];values=data['dt'][:,near].abs()[observed[:,near]]
    save(RUN/'dense_ready.json',dict(complete=True,windows=len(data['roles']),observations=len(records),roles=dict(collections.Counter(data['roles'])),real_near_interval_median_s=float(values.median()),real_near_interval_p90_s=float(values.quantile(.9)),context_frame_offsets=OFFSETS,center_stride=5,target_camera_roundtrip_max_m=error,rgb_native_path=str(RUN/'fresh_dense.pt'),labels_for_training_only=True,scope='Existing training/development clips; detector outputs and GT-free tracking generated for all150real frames per clip; comparisons require coverage checks against sparse centers'))
    print((RUN/'dense_ready.json').read_text(),flush=True)

def main():
    torch.set_num_threads(4);clips=freeze();prepare_observations();assemble_roles_and_targets(clips)

if __name__=='__main__':main()
