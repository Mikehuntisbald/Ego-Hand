"""Actual tiny candidate paths: global optimum, separation and native export."""
import hashlib,json,sys
from pathlib import Path
import numpy as np,torch
from sequence_solver import solve,exhaustive,segments,select_all
R=Path('/mnt/why/native_candidate_sequence_v10_20261019');sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src')
from native_mano import NativeMANO
def main():
    torch.set_num_threads(4);assert json.loads((R/'pilot_frozen.json').read_text())['complete'];p=R/'pilot/clip-002108/candidates.npz';value=np.load(p);j=value['joints_camera21'].astype(np.float64);relative=j-j[:,:,:1];world=np.einsum('nij,nklj->nkli',value['camera_rotation'],relative)[:,:,1:];parts=segments(value['track_id'],value['head_segment'],value['right'],value['timestamp_ns']);ids=next(g[:4] for g in parts if len(g)>=4)
    unary=(value['RGB_cost']+value['prior_cost']).astype(np.float64)[ids,:3];x=world[ids,:3];times=value['timestamp_ns'][ids];path,cost=solve(unary,x,times);truth,true_cost=exhaustive(unary,x,times);assert abs(cost-true_cost)<1e-12 and np.array_equal(path,truth),(cost,true_cost,path,truth)
    # Cut only the temporal graph, never an image or training sample.
    gap=times.copy();gap[2:]+=200_000_000;track=np.zeros(4,int);segment=np.zeros(4,int);side=np.ones(4,bool);groups=segments(track,segment,side,gap);assert [len(g) for g in groups]==[2,2]
    chosen,records=select_all(unary,x,gap,track,segment,side)
    assert np.array_equal(chosen,np.r_[solve(unary[:2],x[:2],gap[:2])[0],solve(unary[2:],x[2:],gap[2:])[0]])
    reset=segment.copy();reset[2:]=1;assert len(segments(track,reset,side,times))==2;chirality=side.copy();chirality[2:]=False;assert len(segments(track,segment,chirality,times))==2
    dt=np.diff(times)/1e9;v0=(x[1,1]-x[0,0])/(60*dt[0]);v1=(x[2,2]-x[1,1])/(60*dt[1]);a=(v1-v0)/(60*(dt[0]+dt[1])/2)
    assert np.isfinite(a).all()
    codec=NativeMANO(device='cpu');fields=['global_orient','hand_pose','betas','transl','reflection'];max_error=0.
    with torch.no_grad():
        for k in range(6):
            native={n:torch.tensor(value[n][:,k]) for n in fields};decoded=codec.decode_exported(native);max_error=max(max_error,float(np.abs(decoded.numpy()-j[:,k]).max()))
    assert max_error<1e-6
    spread=np.linalg.norm(j[:,2:]-j[:,:1],axis=-1)*1000
    out=dict(complete=True,actual_tiny_global_optimum=True,DP_cost=cost,exhaustive_cost=true_cost,paths=3**4,frames=4,candidates=3,gap_reset_chirality_separation=True,real_time_pair_and_triple_units=True,native_export_max_m=max_error,draw_to_mean_distance_mm=dict(mean=float(spread.mean()),P95=float(np.quantile(spread,.95))),no_hand_GT_used=True,production_admitted=False)
    (R/'receipts/preflight.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
