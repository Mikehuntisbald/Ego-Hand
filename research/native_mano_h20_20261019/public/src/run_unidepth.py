"""Real public UniDepth fallback for stereo-rejected observations only."""
import json, sys, time, hashlib
from pathlib import Path
import numpy as np
import torch
import cv2
import os
ROOT=Path(os.environ.get('PUBLIC_HEADHAND_ROOT','/mnt/why/public_headhand_h20_20261019'))
sys.path.insert(0,str(ROOT/'vendor/UniDepth'))
sys.path.insert(0,str(ROOT/'vendor/Ego-Hand/research'))
from unidepth.models import UniDepthV2
from unidepth.utils.camera import Fisheye624
from safetensors.torch import load_file
from geometry import load_frames, camera
from run_depth import surface_intersections, unproject

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    output=ROOT/'outputs/depth_fused_clip2102';output.mkdir(exist_ok=False)
    source=json.loads((ROOT/'outputs/depth_clip2102/depth.json').read_text())
    rows=json.loads((ROOT/'outputs/wilor_clip2102/observations.json').read_text())['frames']
    rf=json.loads((ROOT/'outputs/rf_clip2102/predictions.json').read_text())['frames']
    frames=load_frames(ROOT/'data/clip-002102');surface=np.load(ROOT/'outputs/wilor_clip2102/surface.npz')
    weights=ROOT/'assets/unidepth_public'
    assert hashlib.sha256((weights/'model.safetensors').read_bytes()).hexdigest()=='93705cb3295dd7476b44911b8a55f5215bf74e8d5eccd27cecdb1b338270a648'
    model=UniDepthV2(json.loads((weights/'config.json').read_text())).eval().cuda()
    model.load_state_dict(load_file(str(weights/'model.safetensors')),strict=True)
    model.resolution_level=1
    pending={x['frame_index'] for x in source['observations'] if not x['valid']}
    start=time.time();calls=0;recovered=0
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    try:
        # Ray and Fourier construction use IEEE precision; restore caller after inference.
        torch.backends.cuda.matmul.allow_tf32=False
        for fi in sorted(pending):
            frame=frames[fi];cal=frame['cameras']['214-1']['calibration'];native=camera(cal)
            params=torch.tensor([*native.f,*native.c,*native.distort],device='cuda').float()[None]
            rgb=torch.from_numpy(cv2.cvtColor(cv2.imread(frame['images']['214-1']),cv2.COLOR_BGR2RGB).transpose(2,0,1).copy()).cuda()
            prediction=model.infer(rgb,Fisheye624(params=params));calls+=1
            # Explicit camera-Z output, not radial range.
            depth=prediction['depth'][0,0].float().cpu().numpy()
            with np.load(rf[fi]['output']) as masks:mask_array=masks['masks']
            exclusive=mask_array & ~(mask_array.sum(0)>1)[None]
            for anchor in source['observations']:
                if anchor['frame_index']!=fi or anchor['valid']:continue
                i=anchor['observation'];mask=exclusive[rows[i]['detection']]
                locations=np.flatnonzero(mask & np.isfinite(depth) & (depth>.08)&(depth<3))
                locations=locations[np.linspace(0,len(locations)-1,min(192,len(locations)),dtype=int)] if len(locations) else locations
                y,x=np.unravel_index(locations,depth.shape)
                rays=np.stack([unproject(native,uv) for uv in np.stack([x,y],-1)])
                rays=rays/rays[:,2:3]
                hits=surface_intersections(rays,surface['vertices'][i],surface['faces'])
                values=depth[y,x]-(hits-surface['joints'][i,0,2])
                values=values[np.isfinite(values)&(values>.08)&(values<3)]
                median=float(np.median(values)) if len(values) else None
                mad=float(np.median(np.abs(values-median))) if len(values) else None
                anchor['stereo_failure']=dict(anchor)
                anchor.update(valid=bool(len(values)>=6 and mad<.03),samples=len(values),MAD_m=mad,
                              wrist_depth_m=median if len(values)>=6 and mad<.03 else None,
                              source='public-UniDepthV2-vits14-native-fisheye-surface-corrected')
                recovered+=int(anchor['valid'])
            print(json.dumps(dict(frame=fi,calls=calls,recovered=recovered)),flush=True)
    finally:torch.backends.cuda.matmul.allow_tf32=old_tf32
    source.update(unidepth_calls=calls,unidepth_recovered=recovered,unidepth_seconds=time.time()-start,
                  valid=sum(x['valid'] for x in source['observations']),GT_used=False,
                  unidepth_implementation='public PyTorch FP16; IEEE geometry; author weights; resolution_level=1')
    path=output/'depth.json';path.write_text(json.dumps(source,indent=2))
    (output/'seal.json').write_text(json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),GT_used=False)))
    print(json.dumps(dict(complete=True,valid=source['valid'],total=source['total'])),flush=True)

if __name__=='__main__':main()
