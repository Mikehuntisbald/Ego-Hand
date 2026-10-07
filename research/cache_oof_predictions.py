"""OOF pose predictions with a common RGB basis; no trained fold mixes held GT."""
import json,time
import torch
from torch.utils.data import DataLoader
from train_coarse_pose import PoseDataset
from coarse_pose3d import CoarsePose3D
from oof_common import OLD,RUN,SUBJECTS,save,sha

def main():
    torch.set_num_threads(4);device='cuda:0';protocol=json.loads((RUN/'protocol.json').read_text())
    if (RUN/'cache_done.json').exists():return
    while not (RUN/'fresh_done.json').exists():time.sleep(15)
    for s in SUBJECTS:
        while not (RUN/'folds'/s/'fine/done.json').exists():time.sleep(15)
    rows=[json.loads(l) for l in (OLD/'predicted_manifest.jsonl').read_text().splitlines()]
    for r in rows:
        if r['subject'] in SUBJECTS:r['role']='gate_fit' if r['sequence'] in protocol['gate_fit_sequences'] else 'denoise'
        elif r['subject']=='P0003':r['role']='calibrate' if r['sequence']==protocol['probability_calibration_sequence'] else 'select'
        else:r['role']='development'
    rows += [json.loads(l) for l in (RUN/'fresh_predicted_manifest.jsonl').read_text().splitlines()]
    manifest=RUN/'all_manifest.jsonl';manifest.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
    ds=PoseDataset(manifest);loader=DataLoader(ds,batch_size=256,num_workers=4,pin_memory=True,persistent_workers=True)
    # The fixed, shared projection comes from a seeded untrained pose adapter on
    # pretrained COCO visual layers. No hand 3D labels train these RGB channels.
    torch.manual_seed(20261003);common=CoarsePose3D().to(device).eval()
    data=dict(rows=rows,gt=torch.tensor([r['xyz_camera_m'] for r in rows]),rgb=[])
    with torch.inference_mode():
        for b in loader:
            with torch.autocast('cuda',dtype=torch.bfloat16):p=common(b['image'].to(device),b['geometry'].to(device))
            data['rgb'].append(p['rgb_tokens'].half().cpu())
    data['rgb']=torch.cat(data['rgb']);save(RUN/'rgb_basis.json',dict(initialization='COCO plus seeded frozen projection',seed=20261003,uses_hand_GT=False))
    del common;torch.cuda.empty_cache()
    data['coarse']=torch.empty(len(rows),20,3);data['confidence']=torch.empty(len(rows),21)
    data['in_subject_coarse']=torch.empty(len(rows),20,3);data['in_subject_confidence']=torch.empty(len(rows),21)
    final=CoarsePose3D().to(device);final.load_state_dict(torch.load(OLD/'coarse_fine/best.pt',map_location='cpu',weights_only=False)['model']);final.eval();v1_rgb=[]
    with torch.inference_mode():
        for b in loader:
            ix=b['index']
            with torch.autocast('cuda',dtype=torch.bfloat16):p=final(b['image'].to(device),b['geometry'].to(device))
            data['in_subject_coarse'][ix]=p['xyz'].float().cpu()
            data['in_subject_confidence'][ix]=torch.cat([p['root_confidence'][:,None],p['confidence']],1).float().cpu()
            v1_rgb.append(p['rgb_tokens'].half().cpu())
    data['v1_rgb']=torch.cat(v1_rgb) # Historical comparator only, never v2 training inputs.
    data['coarse'].copy_(data['in_subject_coarse']);data['confidence'].copy_(data['in_subject_confidence'])
    del final;torch.cuda.empty_cache();fold_receipts=[]
    for subject in SUBJECTS:
        path=RUN/'folds'/subject/'fine/best.pt';ck=torch.load(path,map_location='cpu',weights_only=False)
        assert ck['provenance']['held_subject']==subject and subject not in ck['provenance']['trained_subjects']
        ids=[i for i,r in enumerate(rows) if r['subject']==subject]
        model=CoarsePose3D().to(device);model.load_state_dict(ck['model']);model.eval()
        subset=torch.utils.data.Subset(ds,ids);fold_loader=DataLoader(subset,batch_size=256,num_workers=4,pin_memory=True,persistent_workers=True)
        with torch.inference_mode():
            for b in fold_loader:
                ix=b['index']
                with torch.autocast('cuda',dtype=torch.bfloat16):p=model(b['image'].to(device),b['geometry'].to(device))
                data['coarse'][ix]=p['xyz'].float().cpu();data['confidence'][ix]=torch.cat([p['root_confidence'][:,None],p['confidence']],1).float().cpu()
        fold_receipts.append(dict(subject=subject,samples=len(ids),epoch=ck['epoch'],checkpoint_sha256=sha(path),trained_subjects=ck['provenance']['trained_subjects']))
        del model;torch.cuda.empty_cache();print(json.dumps(fold_receipts[-1]),flush=True)
    for key in ['coarse','confidence','rgb','in_subject_coarse','in_subject_confidence']:assert torch.isfinite(data[key]).all(),key
    torch.save(data,RUN/'oof_cache.pt')
    save(RUN/'cache_done.json',dict(complete=True,rows=len(rows),by_role={k:sum(r['role']==k for r in rows) for k in sorted({r['role'] for r in rows})},folds=fold_receipts))
    print((RUN/'cache_done.json').read_text(),flush=True)
if __name__=='__main__':main()
