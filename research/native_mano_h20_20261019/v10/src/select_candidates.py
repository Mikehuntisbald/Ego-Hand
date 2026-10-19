"""GT-free per-frame/RGB and exact whole-track second-order controls."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
from sequence_solver import select_all
R=Path('/mnt/why/native_candidate_sequence_v10_20261019')
ARMS=['frame_RGB','temporal_noRGB','temporal_RGB']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forbid(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=str(args[0]);assert not any(x in name for x in ['.hands.json','__hand_shapes.json__','native_targets.pt','/targets.pt','scoring_arrays.npz','evaluation_labels.pt']),name
def main():
    sys.addaudithook(forbid);assert json.loads((R/'receipts/preflight.json').read_text())['complete'];frozen=json.loads((R/'candidates_frozen.json').read_text());seals={};fields=['state_camera51','joints_camera21','normalized_residual','protection_alpha','global_orient','hand_pose','betas','transl','reflection','wrist_camera','root_rotation_physical'];metadata=['track_id','timestamp_ns','frame_index','right']
    for name,h in frozen['seals'].items():
        seal=R/name;assert sha(seal)==h
        for file,digest in json.loads(seal.read_text())['files'].items():assert sha(seal.parent/file)==digest
        value=np.load(seal.parent/'candidates.npz');j=value['joints_camera21'].astype(np.float64);relative=j-j[:,:,:1];world=np.einsum('nij,nklj->nkli',value['camera_rotation'],relative)[:,:,1:];N=len(j);folder=R/'predictions'/seal.parent.name;folder.mkdir(parents=True,exist_ok=False);reports={}
        for arm in ARMS:
            unary=value['prior_cost'].astype(np.float64).copy()
            if arm!='temporal_noRGB':unary+=value['RGB_cost']
            chosen,parts=select_all(unary,world,value['timestamp_ns'],value['track_id'],value['head_segment'],value['right'],with_motion=arm!='frame_RGB');output={n:value[n][np.arange(N),chosen] for n in fields};output.update({n:value[n] for n in metadata});output['candidate_index']=chosen
            assert np.array_equal(output['wrist_camera'],value['wrist_camera'][:,0]);assert np.array_equal(output['betas'],value['betas'][:,0]);np.savez_compressed(folder/f'{arm}.npz',**output)
            reports[arm]=dict(segments=parts,candidate_counts=np.bincount(chosen,minlength=6).tolist(),switches=sum(p['candidate_switches'] for p in parts),unary_mean=float(unary[np.arange(N),chosen].mean()),GT_selection=False)
        payload=dict(complete=True,GT_used=False,files={p.name:sha(p) for p in folder.glob('*.npz')},arms=reports,source_candidates_sha256=sha(seal.parent/'candidates.npz'));(folder/'seal.json').write_text(json.dumps(payload,indent=2));seals[str((folder/'seal.json').relative_to(R))]=sha(folder/'seal.json');print(json.dumps(dict(clip=folder.name,arms={a:reports[a]['candidate_counts'] for a in ARMS})),flush=True)
    (R/'predictions_frozen.json').write_text(json.dumps(dict(complete=True,freeze_unix=time.time(),seals=seals,candidates_frozen_sha256=sha(R/'candidates_frozen.json'),GT_used=False,source_sha256={p.name:sha(p) for p in (R/'src').glob('*.py')},production_admitted=False),indent=2))
if __name__=='__main__':main()
