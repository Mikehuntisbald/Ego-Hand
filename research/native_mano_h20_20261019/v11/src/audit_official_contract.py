"""Right-only official HOT3D MANO parameter convention, no fabricated LEFT."""
import collections,hashlib,json,re,sys,tarfile
from pathlib import Path
import numpy as np,torch,smplx
R=Path('/mnt/why/native_candidate_risk_v11_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019')
sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src');sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/src')
from native_mano import NativeMANO,common
from geometry import transform
from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);assert json.loads((R/'data/inputs_frozen.json').read_text())['complete'];raw=torch.load(V7/'data/inputs.pt',map_location='cpu',weights_only=False);target=torch.load(V7/'data/native_targets.pt',map_location='cpu',weights_only=False);groups=collections.defaultdict(list)
    for j,i in enumerate(target['indices']):
        row=raw['rows'][int(i)]
        if row['role']=='train' and len(groups[row['subject']])<2:groups[row['subject']].append((j,int(i)))
    records=[]
    for person,items in sorted(groups.items()):
        for j,i in items:
            row=raw['rows'][i];seq=row['sequence'];assert re.fullmatch(r'P[0-9]{4}_[0-9a-f]{8}',seq),'Unexpected recording identifier';clip=int(row['clip']);fi=int(row['frame']);archives=list(Path('/mnt/why/HOT3D/rgb_clips').glob(f'*/{seq}/clip-{clip:06}.tar'));assert len(archives)==1
            with tarfile.open(archives[0]) as tar:h=json.load(tar.extractfile(f'{fi:06}.hands.json'))['right']['mano_pose'];beta=json.load(tar.extractfile('__hand_shapes.json__'))['mano'];pose=transform(json.load(tar.extractfile(f'{fi:06}.cameras.json'))['214-1']['T_world_from_camera'])
            records.append((j,h,beta,pose))
    assert len(records)==12;asset=common.RUN/'assets/MANO_RIGHT.pkl';official=MANOHandModel();official.mano_layer_right=smplx.create(str(asset),'mano',use_pca=True,is_rhand=True,num_pca_comps=15).eval();codec=NativeMANO(device='cpu');tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32)
    with torch.no_grad():
        mesh,landmarks=official(tensor([r[2] for r in records]),tensor([r[1]['thetas'] for r in records]),tensor([r[1]['wrist_xform'] for r in records]),torch.ones(12,dtype=torch.bool));Rs=np.stack([r[3][:3,:3] for r in records]);Ts=np.stack([r[3][:3,3] for r in records]);mesh_cam=np.einsum('nji,nkj->nki',Rs,mesh.numpy()-Ts[:,None]);landmarks_cam=np.einsum('nji,nkj->nki',Rs,landmarks.numpy()-Ts[:,None]);ix=[r[0] for r in records];joints,ours=codec.decode(target['gt'][ix,0],target['root_rotation'][ix],target['local_rotation'][ix],target['beta_GT'][ix],torch.ones(12,dtype=torch.bool),vertices=True)
    mesh_error=float(np.max(np.abs(mesh_cam-ours.numpy())));wrist_error=float(np.max(np.abs(landmarks_cam[:,0]-target['gt'][ix,0].numpy())));assert mesh_error<1e-6 and wrist_error<1e-6
    mapping=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20];landmark_error=np.linalg.norm(landmarks_cam[:,mapping]-joints.numpy(),axis=-1)*1000
    report=dict(complete=True,train_only_samples=12,subjects=6,right_only=True,official_wrapper_source_sha256=sha(Path('/mnt/why/HOT3D-hand-tracking-toolkit/hand_tracking_toolkit/hand_models/mano_hand_model.py')),asset_sha256=sha(asset),official_vertices_max_m=mesh_error,official_wrist_max_m=wrist_error,official_vs_WiLoR_landmark_mm=dict(mean=float(landmark_error.mean()),max=float(landmark_error.max())),same_PCA_and_root_translation_convention=True,no_LEFT_model_loaded_or_fabricated=True,limits='Same existing RIGHT asset; original annotation asset version remains independently unverified. Official tip definition can differ from WiLoR21; metric semantics remain explicit.',production_admitted=False)
    (R/'receipts/official_contract.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['limits']},indent=2))
if __name__=='__main__':main()
