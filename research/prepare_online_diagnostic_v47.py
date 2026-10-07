"""GT-free replay observations for the already-evaluated v46 clips."""
import json,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np,torch,cv2
from hand3d_v8_common import V7,save
from online_parameter_model_v47 import RUN
from temporal_window_v44 import ObservationSampler,window_statistics
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from complete_hand_tracks_stable_v42 import fit_seeds
from hand3d_rollout_v8 import project_fisheye624
from offline_rgb_encoder import crop_roi
import spatial_rgb_common as s

SOURCE=V7.parent/'acceleration_validation_v46';OUT=RUN/'diagnostic_v46'
def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(4);cv2.setNumThreads(0);start=time.time()
    if (OUT/'ready.json').exists():return
    allrows=json.loads((SOURCE/'fresh_rows.json').read_text())
    rows=[{k:r[k] for k in ['sequence','subject','clip','frame','image','camera','box','score','track_id','timestamp_ns']} for r in allrows]
    N=len(rows);d=torch.load(SOURCE/'fresh_data.pt',weights_only=False,mmap=True)
    fields=['world','xyz_camera_bank','available_bank','rotation','translation','rays_world','rgb_bank','positions_bank','roi','scores','risk_rgb_bank']
    data={k:d[k] for k in fields};f,dt=ObservationSampler(rows,rows).sample(list(range(1,N+1)))
    cache={};covered=torch.zeros(N,dtype=torch.bool)
    for path in sorted((SOURCE/'tracks').glob('*_inputs.pt')):
        item=torch.load(path,weights_only=False,map_location='cpu');ix=torch.tensor(item['indices']);covered[ix]=True
        for key,value in item['cache'].items():
            if key not in cache:cache[key]=torch.zeros((N,)+value.shape[1:],dtype=value.dtype)
            cache[key][ix]=value
    assert covered.all();params=cache['camera_params'];bankparams=torch.cat([params[:1]*0,params])
    xy=project_fisheye624(data['xyz_camera_bank'],bankparams)/1408
    observed=torch.isfinite(xy).all(-1)&(xy>=0).all(-1)&(xy<1).all(-1)&data['available_bank']
    data.update(feature_ids=f,dt=dt,xy=torch.nan_to_num(xy)[f],observed_2d=observed[f])
    torch.save(data,OUT/'inputs.pt');torch.save(cache,OUT/'cache.pt');save(OUT/'rows.json',rows)
    right=torch.tensor(json.loads((SOURCE/'side_adapter_audit.json').read_text())['predicted_right'],dtype=torch.long)
    right=torch.cat([right[:1]*0,right]);torch.save(right,OUT/'right.pt')
    device='cuda:3';decoder=ParameterTrajectoryDecoder(device);seeds=torch.zeros(N+1,21,3)
    for begin in range(1,N+1,512):
        end=min(begin+512,N+1);seeds[begin:end]=fit_seeds(decoder,data['xyz_camera_bank'][begin:end].to(device),right[begin:end].to(device)).cpu()
    torch.save(seeds,OUT/'coarse.pt');del decoder;torch.cuda.empty_cache()
    pixels=np.lib.format.open_memmap(OUT/'pixels.npy',mode='w+',dtype=np.uint8,shape=(N+1,256,256,3))
    def prep(i):
        r=rows[i];images,pos,*_=s.prepare(r,crop_roi(r['box']),[[0,0,0,0]])
        assert np.max(np.abs(pos-data['positions_bank'][i+1].numpy()))<2e-5
        return i+1,images[0]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for fid,image in pool.map(prep,range(N)):pixels[fid]=image
    pixels.flush();del pixels
    save(OUT/'ready.json',dict(complete=True,observations=N,window_statistics=window_statistics(f,dt),
        uses_gt=False,scope='Replay of previously evaluated v46 clips; no fresh or new-subject claim',
        common_solver='Same whole-batch solver call for every head; historical per-track v46 numbers are not direct training-effect controls',
        common_risk_and_RGB_solver_evidence_frozen=True,seconds=time.time()-start))
    print((OUT/'ready.json').read_text(),flush=True)
if __name__=='__main__':main()
