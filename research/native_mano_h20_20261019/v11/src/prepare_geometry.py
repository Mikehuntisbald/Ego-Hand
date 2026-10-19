"""Prediction/calibration-only crop chart; separate immutable input stage."""
import collections,hashlib,json,re,sys,tarfile
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_candidate_risk_v11_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019')
sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/src');sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src')
from geometry import camera
from native_mano import common
Q=np.array([[0,1,0],[-1,0,0],[0,0,1]],np.float32)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forbid(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=str(args[0]);assert not any(x in path for x in ['.hands.json','__hand_shapes.json__','native_targets.pt','quality_targets.pt','/targets.pt','training_targets.pt']), 'Hand-label access forbidden during geometry input preparation'
def crop_chart(box,cal):
    try:
        cam=camera(cal);box=np.asarray(box,float);center=(box[:2]+box[2:])/2;z=common.unproject(cam,center);z/=np.linalg.norm(z);x=np.array([1.,0.,0.]);x-=z*np.dot(x,z);x/=np.linalg.norm(x);rotation=np.stack([x,np.cross(z,x),z],1)
        corners=np.array([[box[0],box[1]],[box[2],box[1]],[box[2],box[3]],[box[0],box[3]],[center[0],box[1]],[center[0],box[3]],[box[0],center[1]],[box[2],center[1]]]);rays=np.stack([common.unproject(cam,p) for p in corners])@rotation;half=max(float(np.abs(rays[:,:2]/rays[:,2:3].clip(.05)).max()),.015)*1.3
        return (rotation@Q.T).astype(np.float32),float(95.5/half)
    except Exception:raise RuntimeError('Numeric crop calibration contract failed') from None
def main():
    sys.addaudithook(forbid);assert json.loads((R/'data/candidates_ready.json').read_text())['complete'];assert not (R/'data/inputs_frozen.json').exists();bank=torch.load(R/'data/candidates.pt',map_location='cpu',weights_only=False);raw=torch.load(V7/'data/inputs.pt',map_location='cpu',weights_only=False);rows=raw['rows'];groups=collections.defaultdict(list);N=len(bank['indices']);rotation=np.zeros((N,3,3),np.float32);focal=np.zeros(N,np.float32)
    for j,i in enumerate(bank['indices']):
        row=rows[int(i)];seq=row['sequence'];assert isinstance(seq,str) and re.fullmatch(r'P[0-9]{4}_[0-9a-f]{8}',seq),'Unexpected recording identifier';clip=int(row['clip']);assert 0<=clip<1000000;groups[(seq,clip)].append((j,int(i)))
    for (seq,clip),items in sorted(groups.items()):
        archives=list(Path('/mnt/why/HOT3D/rgb_clips').glob(f'*/{seq}/clip-{clip:06}.tar'));assert len(archives)==1,'Expected a unique local archive'
        with tarfile.open(archives[0]) as tar:
            for j,i in items:
                fi=int(rows[i]['frame']);assert 0<=fi<1000000;cal=json.load(tar.extractfile(f'{fi:06}.cameras.json'))['214-1']['calibration'];rotation[j],focal[j]=crop_chart(rows[i]['box'],cal)
    xyz=bank['joints_camera21'].numpy();local=np.einsum('nji,nklj->nkli',rotation,xyz);assert np.isfinite(local).all() and (local[...,2]>.01).all();uv=local[...,:2]/local[...,2:3]*focal[:,None,None,None]+127.5;uv=(uv-np.array([31.5,-.5]))/np.array([192.,256.]);torch.save(dict(indices=bank['indices'],crop_uv=torch.tensor(uv,dtype=torch.float32),rotation=torch.tensor(rotation),focal=torch.tensor(focal)),R/'data/geometry.pt')
    report=dict(complete=True,hand_GT_read=False,observations=N,candidates_each=6,geometry_sha256=sha(R/'data/geometry.pt'),candidate_sha256=sha(R/'data/candidates.pt'),input_metadata_sha256=sha(V7/'data/inputs.pt'),source_sha256=sha(Path(__file__)),production_admitted=False);(R/'data/inputs_frozen.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
