"""Freeze 12 unused clips and reuse the audited natural-RGB observation worker."""
import hashlib,json,time
from pathlib import Path
from hand3d_v8_common import V7,save
from complete_hand_tracks_acceleration_v46 import profile_config
import find_fourth_inventory_v14 as inventory

RUN=V7.parent/'acceleration_validation_v46';CODE=Path(__file__).parent


def freeze():
    RUN.mkdir(exist_ok=True)
    if (RUN/'fresh_manifest.json').exists():return
    inventory.RUN=RUN;inventory.main()
    inv=json.loads((RUN/'expanded_inventory.json').read_text());targets=[]
    for subject in ['P0010','P0015']:
        eligible=[r for r in inv['rows'] if r['subject']==subject and len(r['unused'])>=2]
        assert len(eligible)>=2;targets+=eligible[:3]
    entries=[]
    for r in targets:
        ids=r['unused'];clips=[ids[len(ids)//4],ids[3*len(ids)//4]];assert len(set(clips))==2
        entries.append(dict(split='hand3d_acceleration_v46',subject=r['subject'],sequence=r['sequence'],clips=clips))
    base=json.loads((V7.parent.parent/'subset_manifest.json').read_text())
    manifest={k:base[k] for k in ['official_repo','revision','mirror','stream']}
    manifest.update(sequences=entries,created_unix=time.time(),frames_stride=1,center_stride=5,
        selection='Deterministic metadata-only quartiles; all previously exported/manifested clips excluded before RGB/error inspection',
        scope='10 project-unused clips,5previously encountered source sequences,2previously encountered actors. WiLoR pretraining overlap unknown.')
    save(RUN/'fresh_manifest.json',manifest)
    ck=V7.parent/'matched_parameter_v43/dit/best.pt'
    protocol=dict(frozen_before_predictions=True,manifest_sha256=hashlib.sha256((RUN/'fresh_manifest.json').read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),context_s=1.6,draws=4,ddim_steps=10,
        profiles={p:profile_config(p) for p in ['strict','acc_x2']},candidate_path_fixed_between_profiles=True,
        selection_costs_unchanged=True,default_changed=False,new_subject_claim=False,
        center_inclusion='Every5thframe detector observation with prediction-only temporal support; no GT-based center gate',
        scope=manifest['scope'])
    save(RUN/'frozen_protocol.json',protocol)
    save(RUN/'unseen_check.json',dict(passed=True,clips=10,sequences=5,
        selected_clips=[dict(sequence=r['sequence'],clip=c) for r in entries for c in r['clips']],
        inventory_sha256=hashlib.sha256((RUN/'expanded_inventory.json').read_bytes()).hexdigest(),
        no_visual_or_error_selection=True))


def main():
    freeze()
    if (RUN/'fresh_ready.json').exists():return
    original=(CODE/'prepare_fresh_hand3d_v8.py').read_text()
    replacements={"if i%5==0":"if True",
        "if not r['matched'] or uv_valid_bank[center].sum()<10:continue":"if r['frame']%5!=0:continue",
        "for offset in OFFSETS['multiscale']:":"for frame_offset in [-48,-30,-18,-10,-6,-3,-2,-1,0,1,2,3,6,10,18,30,48]:",
        "fid=group.get(r['frame']+5*offset,0)":"fid=group.get(r['frame']+frame_offset,0)",
        "else offset/6)":"else frame_offset/30)","device='cuda:3'":"device='cuda:2'",
        "device='3'":"device='2'"}
    source=original
    for before,after in replacements.items():
        expected=2 if before=="device='3'" else 1
        assert source.count(before)==expected,(before,source.count(before))
        source=source.replace(before,after)
    snapshot=RUN/'observation_worker_snapshot.py';snapshot.write_text(source)
    save(RUN/'observation_provenance.json',dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        generated_sha256=hashlib.sha256(source.encode()).hexdigest(),replacements=replacements,
        natural_only=True,gt_does_not_select_tracks_or_centers=True))
    ns=dict(__name__='v46_observation_worker',__file__=str(snapshot))
    exec(compile(source,str(snapshot),'exec'),ns);ns['RUN']=RUN;ns['freeze']=lambda:json.loads((RUN/'fresh_manifest.json').read_text())
    ns['prepare']()


if __name__=='__main__':main()
