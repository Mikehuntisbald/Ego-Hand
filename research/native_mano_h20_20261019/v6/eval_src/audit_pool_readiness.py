"""Metadata-only plan for expanded training; no labels or GPU work."""
import collections,json,sys
from pathlib import Path
import torch
R=Path('/mnt/why/native_mano_shape_v6_20261019')
sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/vendor/Ego-Hand/research')
from prepare_joint_mano_v28 import records_bank
def main():
    d=torch.load('/mnt/why/HOT3D/experiments/online_rgb_3d_v47/inputs.pt',weights_only=False,mmap=True,map_location='cpu')
    indices=[i for i,role in enumerate(d['roles']) if role=='train'];_,records=records_bank(d)
    fids=torch.unique(d['feature_ids'][indices]).tolist();fids=[int(i) for i in fids if i>0]
    clips=collections.defaultdict(list)
    for i in indices:
        row=d['rows'][i];clips[(d['subjects'][i],row['sequence'],int(row['clip']))].append(i)
    available=[]
    for (subject,sequence,clip),ids in sorted(clips.items()):
        fid=int(d['feature_ids'][ids[0],8]);record=records[fid-1];archives=list(Path('/mnt/why/HOT3D/rgb_clips').glob(f'*/{sequence}/clip-{clip:06}.tar'))
        available.append(dict(subject=subject,sequence=sequence,clip=clip,windows=len(ids),representative_RGB_exists=Path(record['image']).is_file(),archive=str(archives[0]) if len(archives)==1 else None,archive_matches=len(archives)))
    report=dict(complete=True,train_windows=len(indices),train_clips=len(available),train_persons=sorted({k[0] for k in clips}),train_context_observations=len(fids),
        estimated_full_spatial_RGB_cache_FP16_bytes=(len(fids)+1)*192*1280*2,clips=available,
        scope='only metadata and representative file checks, not a claim all pixels/labels are validated; next data build must verify every input',
        no_hand_labels_read=True,no_training_started=True,not_fresh_generalization=True)
    (R/'receipts/full_training_pool_ready.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
