"""Loss-only native MANO projections in the exact rotated WiLoR image band."""
import collections,hashlib,json,sys,tarfile
from pathlib import Path
import cv2,numpy as np,torch
R=Path('/mnt/why/native_rgb_observation_v9_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019')
sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src');sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/src')
from native_mano import NativeMANO,common
from geometry import camera,IDENTITY
Q=np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def crop_geometry(box,cal):
    cam=camera(cal);box=np.asarray(box,float);center=(box[:2]+box[2:])/2;ray=common.unproject(cam,center);z=ray/np.linalg.norm(ray);x=np.array([1.,0.,0.]);x-=z*np.dot(x,z);x/=np.linalg.norm(x);y=np.cross(z,x);rotation=np.stack([x,y,z],axis=1)
    corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]],[center[0],box[1]],[center[0],box[3]],[box[0],center[1]],[box[2],center[1]]]);rays=np.stack([common.unproject(cam,p) for p in corners])@rotation;xy=rays[:,:2]/rays[:,2:3].clip(.05);focal=95.5/(max(float(np.abs(xy).max()),.015)*1.3)
    return (rotation@Q.T).astype(np.float32),float(focal)
def project_crop(xyz,rotation,focal):
    local=xyz@rotation;uv=local[...,:2]/local[...,2:3]*focal+127.5
    # First network pixel center is x=32,y=0; use align_corners=False centers.
    return (uv-np.array([31.5,-.5]))/np.array([192.,256.])
def main():
    torch.set_num_threads(4);cv2.setNumThreads(1);(R/'data').mkdir(parents=True,exist_ok=True);(R/'receipts').mkdir(exist_ok=True)
    assert not (R/'data/ready.json').exists();raw=torch.load(V7/'data/inputs.pt',map_location='cpu',weights_only=False);targets=torch.load(V7/'data/native_targets.pt',map_location='cpu',weights_only=False);rows=raw['rows'];ids=targets['indices'].tolist();fids=raw['feature_ids'][ids,8]
    assert len(ids)==1355 and bool(raw['right'][fids].bool().all());codec=NativeMANO(device='cpu')
    with torch.no_grad():coarse=codec.from_state(raw['state'][fids],raw['betas'][fids],raw['right'][fids]).numpy()
    groups=collections.defaultdict(list)
    for j,i in enumerate(ids):groups[(rows[i]['sequence'],int(rows[i]['clip']))].append((j,i))
    rotation=np.zeros((len(ids),3,3),np.float32);focal=np.zeros(len(ids),np.float32);truth=np.zeros((len(ids),21,2),np.float32);baseline=np.zeros_like(truth);original_uv=np.zeros_like(truth);roundtrip=0.;crop_parity=0.;pixel_checks=0
    crops=np.load(V7/'data/crops.npy',mmap_mode='r');chosen=set(np.linspace(0,len(ids)-1,64,dtype=int).tolist())
    for (seq,clip),items in sorted(groups.items()):
        archives=list(Path('/mnt/why/HOT3D/rgb_clips').glob(f'*/{seq}/clip-{clip:06}.tar'));assert len(archives)==1
        with tarfile.open(archives[0]) as tar:
            for j,i in items:
                row=rows[i];fi=int(row['frame']);cal=json.load(tar.extractfile(f'{fi:06}.cameras.json'))['214-1']['calibration'];rot,f=crop_geometry(row['box'],cal);gt=targets['gt'][j].numpy();truth[j]=project_crop(gt,rot,f);baseline[j]=project_crop(coarse[j],rot,f);original_uv[j]=camera(cal).eye_to_window(gt);rotation[j]=rot;focal[j]=f
                uv=truth[j]*np.array([192.,256.])+np.array([31.5,-.5]);local=np.concatenate([(uv-127.5)/f,np.ones((21,1))],1);ray=local@rot.T;source=gt/np.linalg.norm(gt,axis=-1,keepdims=True);back=ray/np.linalg.norm(ray,axis=-1,keepdims=True);err=float(np.max(np.abs(source-back)));roundtrip=max(roundtrip,err);assert err<2e-6
                if j in chosen:
                    image=cv2.imdecode(np.frombuffer(tar.extractfile(f'{fi:06}.image_214-1.jpg').read(),np.uint8),cv2.IMREAD_COLOR);crop,r0,f0,_=common.crop(image,row['box'],dict(calibration=cal,T_world_from_camera=IDENTITY),padding=1.3)
                    crop_parity=max(crop_parity,float(np.max(np.abs(rot-r0@Q.T))),abs(f-f0));assert np.array_equal(crop,crops[int(fids[j])]),('source crop mismatch',j);pixel_checks+=1
    valid=np.isfinite(truth).all(-1);inband=valid&(truth>=0).all(-1)&(truth<=1).all(-1);native_image=(original_uv>=0).all(-1)&(original_uv<1408).all(-1)
    pack=dict(indices=targets['indices'],feature_ids=fids,uv=torch.from_numpy(truth),coarse_uv=torch.from_numpy(baseline),finite=torch.from_numpy(valid),inband=torch.from_numpy(inband),native_image=torch.from_numpy(native_image),crop_rotations=torch.from_numpy(rotation),crop_focals=torch.from_numpy(focal))
    torch.save(pack,R/'data/targets.pt');ready=dict(complete=True,train=1330,dev_select=25,right_only=True,GT_scope='Native MANO projected amodal locations; no observed-joint or visibility annotations. inband means image-band framing, not visibility.',source_native_targets_sha256=sha(V7/'data/native_targets.pt'),source_inputs_sha256=sha(V7/'data/inputs.pt'),target_sha256=sha(R/'data/targets.pt'),source_sha256=sha(Path(__file__)),ray_roundtrip_max=roundtrip,crop_geometry_parity_max=crop_parity,source_pixel_exact_checks=pixel_checks,inband_points=int(inband.sum()),all_points=int(inband.size),native_image_points=int(native_image.sum()),GT_inference=False,production_admitted=False)
    (R/'data/ready.json').write_text(json.dumps(ready,indent=2));print(json.dumps(ready,indent=2))
if __name__=='__main__':main()
