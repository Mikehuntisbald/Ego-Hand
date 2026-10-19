"""Check delivered parameter/JSON/H5 geometry, then freeze before label access."""
import argparse, json, hashlib, sys, time, tarfile
from pathlib import Path
import numpy as np
import torch
import os
ROOT=Path(os.environ.get('PUBLIC_HEADHAND_ROOT','/mnt/why/public_headhand_h20_20261019'))
sys.path.insert(0,str(ROOT/'vendor/Ego-Hand/research'))
import wilor_eval_common
from mano_model import load_decoder
from geometry import load_frames

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--suffix',default='');args=parser.parse_args()
    torch.set_num_threads(4);decoder=load_decoder();frames=load_frames(ROOT/'data/clip-002102')
    reports={};hashes={}
    for name in [f'mano_baseline_clip2102{args.suffix}',f'mano_dit_clip2102{args.suffix}']:
        folder=ROOT/'outputs'/name;seal=json.loads((folder/'seal.json').read_text())
        for fname,expected in seal['files'].items():assert hashlib.sha256((folder/fname).read_bytes()).hexdigest()==expected
        z=np.load(folder/'parameters.npz');data=json.loads((folder/'joints3d_world.json').read_text())
        with torch.no_grad():
            joints=decoder(torch.tensor(z['state_world51'],device='cuda'),torch.tensor(z['betas_shared'][z['group']],device='cuda'),torch.tensor(z['right'],device='cuda')).cpu().numpy()
        fk=float(np.max(np.abs(joints-z['joints_world'])));assert fk<1e-6
        assert len(data['frames'])==len(frames)
        for a,b in zip(data['frames'],frames):assert a['timestamp_ns']==b['timestamp_ns']
        world=z['joints_world'];camera_xyz=z['joints_camera'];errors=[]
        for i,fi in enumerate(z['frame_index']):
            f=data['frames'][fi];h=next(x for x in f['hands'] if x['track_id']==int(z['track_id'][i]))
            errors.append(np.max(np.abs(world[i]-h['joints_world_m'])))
            T=np.array(f['T_world_from_camera']);errors.append(np.max(np.abs((camera_xyz[i]@T[:3,:3].T+T[:3,3])-world[i])))
        assert max(errors)<1e-6
        import h5py
        with h5py.File(folder/'bundle.h5') as f:
            assert np.array_equal(f['timestamp_ns'][:],[r['timestamp_ns'] for r in frames])
            for tid in np.unique(z['track_id']):
                ids=np.flatnonzero(z['track_id']==tid);assert np.allclose(f[f'track_{tid}/joints_world_m'][:][z['frame_index'][ids]],world[ids])
        reports[name]=dict(FK_max_m=fk,coordinate_JSON_max_m=float(max(errors)),timeline_frames=len(frames),H5_verified=True,
                           finite=True,GT_used=False)
    for path in sorted((ROOT/'outputs').glob('*/seal.json')):
        if 'failed' not in str(path):hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    snapshot=ROOT/f'receipts/inference_sources_frozen{args.suffix}.tar.gz'
    with tarfile.open(snapshot,'w:gz') as tar:
        for p in sorted((ROOT/'src').glob('*.py')):tar.add(p,arcname=p.name)
    replay=bool(args.suffix) or os.environ.get('PUBLIC_DEVELOPMENT_REPLAY')=='1'
    report=dict(complete=True,freeze_unix=time.time(),GT_read_before_freeze=replay,current_predictions_scored_only_after_freeze=True,reports=reports,seals=hashes,
                sources_sha256=hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                scope='development repair on reused clip; no held-out claim' if replay else 'one diagnostic public clip; training overlap not yet audited; no generalization claim',
                development_labels_opened_previously=replay)
    path=ROOT/f'receipts/predictions_frozen_before_GT{args.suffix}.json';assert not path.exists();path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
