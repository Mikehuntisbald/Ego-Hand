"""Immutable five-source stage B; no test XML is parsed here."""
import json,hashlib,sys
from pathlib import Path
from PIL import Image
import prepare_rfdetr_multidata_v53 as exporter
from acquire_surgical_images_v51 import split

A=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53_20261007')
B=Path('/mnt/why/HOT3D/experiments/rfdetr_multidata_v53B_20261007')

def main():
    assert (A/'100doh_done.json').exists(), 'Wait for verified 100DOH acquisition'
    assert not (B/'dataset_protocol.json').exists(), 'Never overwrite an existing frozen dataset'
    B.mkdir(exist_ok=True)
    rows=list(map(json.loads,(A/'domain_records.jsonl').read_text().splitlines()))
    original={r['id'] for r in rows}
    surg=Path('/mnt/why/HOT3D/domain_data_v51/surgical_hands')
    annotations=json.loads((surg/'annotations.json').read_text())
    index={e['name']:e for e in map(json.loads,(surg/'tar_index.jsonl').read_text().splitlines())}
    extra=[]
    for clip,data in annotations.items():
        group=clip[:11]
        if split(group)!='train':continue
        by_id={}
        for ann in data['annotations']:by_id.setdefault(ann['image_id'],[]).append(ann)
        for im in data['images']:
            if not im['is_labeled'] or im['id'] in original:continue
            name=f"surgical_hands_release/images/{im['video_dir']}/{im['file_name']}"
            if name not in index:continue
            paths=[surg/'selected_images'/im['video_dir']/im['file_name'],A/'surgical_images'/im['video_dir']/im['file_name']]
            image=next((p for p in paths if p.exists() and p.stat().st_size==index[name]['size']),None)
            if image is None:continue
            try:
                with Image.open(image) as pic:pic.verify()
            except Exception:continue
            hands=[dict(instance_id=str(v['track_id']),box_xyxy=v['bbox'],has_keypoint_GT=True) for v in by_id.get(im['id'],[])]
            if not hands:continue
            extra.append(dict(id=im['id'],dataset='surgical_hands',group=group,clip=clip,split='train',image=str(image),image_size=[im['frame_width'],im['frame_height']],hands=hands,real_RGB=True,GT_3D=False))
    rows+=extra
    rows+=list(map(json.loads,(A/'100doh_records.jsonl').read_text().splitlines()))
    for f in ['background_review/accepted_training_negatives.jsonl','background_development_review/accepted_development_negatives.jsonl']:
        rows+=list(map(json.loads,(A/f).read_text().splitlines()))
    ids=[v['id'] for v in rows];assert len(ids)==len(set(ids)), 'duplicate records'
    for domain in {v['dataset'] for v in rows}:
        groups={s:{v['group'] for v in rows if v['dataset']==domain and v['split']==s} for s in ['train','dev','test']}
        assert not groups['train']&(groups['dev']|groups['test']) and not groups['dev']&groups['test'],domain
    exporter.DOMAIN['clinical_background_review']=5
    exporter.REPEATS['clinical_background_review']=4
    # Extra surgical frames increase coverage; keep their repeat factor at 2.
    statistics=exporter.export(rows,B/'data')
    # Mask AP must be measured only against actual human mask labels.
    ego_dev=[v for v in rows if v['dataset']=='egohands' and v['split']=='dev']
    exporter.export(ego_dev,B/'ego_validation',False)
    import shutil
    shutil.rmtree(B/'data/valid')
    shutil.copytree(B/'ego_validation/valid',B/'data/valid',symlinks=True)
    pilot=[]
    for domain in exporter.DOMAIN:
        for role,n in [('train',24),('dev',8)]:
            pool=sorted((v for v in rows if v['dataset']==domain and v['split']==role),key=lambda v:hashlib.sha256(('pilot53B:'+v['id']).encode()).hexdigest())
            pilot+=pool[:n]
    exporter.export(pilot,B/'pilot_data',False)
    raw=''.join(json.dumps(v)+'\n' for v in rows)
    (B/'domain_records.jsonl').write_text(raw)
    (B/'fresh_test_rgb_records.jsonl').write_bytes((A/'100doh_test_rgb_records.jsonl').read_bytes())
    hashes={v['id']:hashlib.file_digest(open(v['image'],'rb'),'sha256').hexdigest() for v in rows}
    (B/'training_image_hashes.json').write_text(json.dumps(hashes))
    protocol=dict(statistics=statistics,extra_surgical_verified_train=len(extra),surgical_acquisition_complete=(A/'surgical_expand_done.json').exists(),source_stageA=str(A),record_sha256=hashlib.sha256(raw.encode()).hexdigest(),repeats=exporter.REPEATS,starting_checkpoint=str(A/'full/best_admitted_ema.pth'),starting_checkpoint_not_safe_deployment_admission=True,mask_GT_only_Ego=True,CPPE_unknown_unmatched_queries=True,background_train=50,background_dev=23,background_weak_presence_review=True,test_XML_not_parsed=True,fresh_test_source_groups=400,synthetic_occlusion=False,default_replaced=False)
    (B/'dataset_protocol.json').write_text(json.dumps(protocol,indent=2))
    print(json.dumps(protocol),flush=True)

if __name__=='__main__':main()
