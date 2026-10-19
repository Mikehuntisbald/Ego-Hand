"""Convert image/prior unary units before reading any new scoring labels.

Retain original v10 outputs. Same speed/acceleration weights and scales.
Near Gaussian mode, NLL=e^2/(2*sigma^2); BA20*smoothL1(e/span)
has curvature20/span^2, giving conversion20*(sigma/span)^2.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,'/mnt/why/native_candidate_sequence_v10_20261019/src')
from sequence_solver import select_all,solve,exhaustive
R=Path('/mnt/why/native_candidate_sequence_v10_20261019');REV=R/'likelihood_r1';FACTOR=20*(4/(191.5/1.3))**2
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forbid(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=str(args[0]);assert not any(x in name for x in ['.hands.json','__hand_shapes.json__','native_targets.pt','/targets.pt','scoring_arrays.npz','evaluation_labels.pt']),name
def main():
    sys.addaudithook(forbid);assert json.loads((R/'predictions_frozen.json').read_text())['complete'];assert not (R/'evaluation').exists() and not (R/'full_review').exists(),'No post-scoring retuning';REV.mkdir(exist_ok=False)
    protocol=dict(complete=True,reason='Existing BA motion weights are numerically tiny in raw image heatmap NLL units. Initial original3 arms retained; no new scoring GT read before this revision.',factor=FACTOR,derivation='20*(sigma4crop_pixels / effective_box_span(191.5/1.3))^2; near-mode curvature conversion, approximate perspective chart; not test-GT tuning',unary='Scale BOTH RGB and candidate0 prior by same common factor; framewise choices therefore unchanged',motion='Exact same v=0.01m,a=0.005m and weights0.002/0.02, all real dt; gates unchanged',arms=['temporal_noRGB_scaled','temporal_RGB_scaled'],focus='temporal_RGB_scaled',source_sha256=sha(Path(__file__)),GT_used=False,production_admitted=False)
    (REV/'protocol.json').write_text(json.dumps(protocol,indent=2));frozen=json.loads((R/'candidates_frozen.json').read_text());seals={};frame_parity=True
    fields=['state_camera51','joints_camera21','normalized_residual','protection_alpha','global_orient','hand_pose','betas','transl','reflection','wrist_camera','root_rotation_physical'];metadata=['track_id','timestamp_ns','frame_index','right']
    for name,h in frozen['seals'].items():
        p=R/name;assert sha(p)==h;assert sha(p.parent/'candidates.npz')==json.loads(p.read_text())['files']['candidates.npz'];v=np.load(p.parent/'candidates.npz');j=v['joints_camera21'].astype(np.float64);world=np.einsum('nij,nklj->nkli',v['camera_rotation'],j-j[:,:,:1])[:,:,1:];folder=REV/'predictions'/p.parent.name;folder.mkdir(parents=True);reports={};N=len(j)
        unary=v['prior_cost'].astype(np.float64)+v['RGB_cost'];assert np.array_equal(unary.argmin(1),(unary*FACTOR).argmin(1));assert np.array_equal(unary.argmin(1),np.load(R/'predictions'/p.parent.name/'frame_RGB.npz')['candidate_index'])
        for arm in protocol['arms']:
            unary=v['prior_cost'].astype(np.float64).copy()
            if arm=='temporal_RGB_scaled':unary+=v['RGB_cost']
            unary*=FACTOR;chosen,segments=select_all(unary,world,v['timestamp_ns'],v['track_id'],v['head_segment'],v['right']);result={k:v[k][np.arange(N),chosen] for k in fields};result.update({k:v[k] for k in metadata});result['candidate_index']=chosen;np.savez_compressed(folder/f'{arm}.npz',**result);reports[arm]=dict(candidate_counts=np.bincount(chosen,minlength=6).tolist(),segments=segments,switches=sum(s['candidate_switches'] for s in segments))
            if not seals and arm=='temporal_RGB_scaled':
                ids=np.array([i for i in range(N) if v['track_id'][i]==v['track_id'][0]])[:4];a,cost=solve(unary[ids,:3],world[ids,:3],v['timestamp_ns'][ids]);b,true=exhaustive(unary[ids,:3],world[ids,:3],v['timestamp_ns'][ids]);assert abs(cost-true)<1e-12 and np.array_equal(a,b)
        report=dict(complete=True,GT_used=False,files={p.name:sha(p) for p in folder.glob('*.npz')},arms=reports,framewise_ranking_unchanged=True);(folder/'seal.json').write_text(json.dumps(report,indent=2));seals[str((folder/'seal.json').relative_to(R))]=sha(folder/'seal.json');print(json.dumps(dict(clip=folder.name,arms={a:reports[a]['candidate_counts'] for a in protocol['arms']})),flush=True)
    (REV/'predictions_frozen.json').write_text(json.dumps(dict(complete=True,freeze_unix=time.time(),seals=seals,protocol_sha256=sha(REV/'protocol.json'),GT_used=False),indent=2));(R/'all_predictions_frozen.json').write_text(json.dumps(dict(complete=True,freeze_unix=time.time(),original_sha256=sha(R/'predictions_frozen.json'),revision_sha256=sha(REV/'predictions_frozen.json'),GT_used=False),indent=2))
if __name__=='__main__':main()
