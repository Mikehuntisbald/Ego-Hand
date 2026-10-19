"""Right-hand native parameter targets. No fitting to UmeTrack landmarks."""
import collections,hashlib,json,sys,tarfile
from pathlib import Path
import numpy as np,torch,smplx
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation
R=Path('/mnt/why/native_mano_supervision_v5_20261019')
OLD=Path('/mnt/why/native_mano_dit_20261019')
V2=Path('/mnt/why/native_mano_dit_v2_20261019')
sys.path.insert(0,str(V2/'src'));sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/src')
from native_mano import NativeMANO,exp_so3,common
from geometry import transform

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def iou(a,b):
    lo=np.maximum(a[:2],b[:2]);hi=np.minimum(a[2:],b[2:]);inter=np.maximum(hi-lo,0).prod()
    return inter/max(1e-8,np.maximum(a[2:]-a[:2],0).prod()+np.maximum(b[2:]-b[:2],0).prod()-inter)

def main():
    torch.set_num_threads(4);R.mkdir(parents=True,exist_ok=True);(R/'data').mkdir(exist_ok=True);(R/'receipts').mkdir(exist_ok=True)
    assert not (R/'data/ready.json').exists(),'Immutable completed targets'
    raw=torch.load(OLD/'data/inputs.pt',map_location='cpu',weights_only=False);rows=raw['rows'];print(json.dumps(rows[:2]),flush=True)
    asset=common.RUN/'assets/MANO_RIGHT.pkl'
    protocol=dict(supervision='actual dataset MANO15 PCA, expanded full axis angles; no UmeTrack IK or fitting',side='right-only pilot; left passthrough at deployment',
        input_sha256=sha(OLD/'data/inputs.pt'),asset_sha256=sha(asset),source_script_sha256=sha(Path(__file__)),
        roles='same six train persons and dev_select P0003; dev_calibrate and six validation clips excluded from target preparation',
        matching='one-to-one IoU on all provided GT boxes, >=0.1, then right native GT and predicted-right filter; no error-based filtering',
        fixed_shape=True,fixed_wrist=True,ground_truth_inference_input=False,model='same native51/17-token model, 4-draw10-step mean',
        limits='existing licensed WiLoR RIGHT asset; annotation-original asset version not independently verified; no LEFT asset fabricated',
        selection='600 fixed updates, checkpoint at200/400/600, dev relative20 +20*good-point-harm-fraction; same all arms',natural_RGB_only=True)
    (R/'protocol.json').write_text(json.dumps(protocol,indent=2))
    eligible=[i for i,r in enumerate(rows) if r['role'] in ['train','dev_select']]
    groups=collections.defaultdict(list)
    for i in eligible:groups[(rows[i]['sequence'],int(rows[i]['clip']))].append(i)
    records=[];counts=collections.Counter()
    for (sequence,clip),ids in sorted(groups.items()):
        archives=list(Path('/mnt/why/HOT3D/rgb_clips').glob(f'*/{sequence}/clip-{clip:06}.tar'));assert len(archives)==1,(sequence,clip,archives)
        with tarfile.open(archives[0]) as tar:
            beta=json.load(tar.extractfile('__hand_shapes.json__'))['mano'];byframe=collections.defaultdict(list)
            for i in ids:byframe[int(rows[i]['frame'])].append(i)
            for fi,frameids in sorted(byframe.items()):
                hands=json.load(tar.extractfile(f'{fi:06}.hands.json'));cal=json.load(tar.extractfile(f'{fi:06}.cameras.json'))['214-1'];pose=transform(cal['T_world_from_camera'])
                targets=[(side,h,np.asarray(h['boxes_amodal']['214-1'],float)) for side,h in hands.items() if '214-1' in h.get('boxes_amodal',{})]
                if not targets:counts['no_GT_box']+=len(frameids);continue
                overlap=np.array([[iou(np.asarray(rows[i]['box'],float),x[2]) for x in targets] for i in frameids]);a,b=linear_sum_assignment(1-overlap)
                for x,y in zip(a,b):
                    i=frameids[x];side,h,_=targets[y];fid=int(raw['feature_ids'][i,8]);counts['matched_candidates']+=1
                    if overlap[x,y]<.1:counts['low_IoU']+=1;continue
                    if side!='right':counts['left_GT_excluded']+=1;continue
                    if not bool(raw['right'][fid]):counts['wrong_predicted_side_excluded']+=1;continue
                    if not h.get('mano_pose'):counts['missing_MANO']+=1;continue
                    records.append(dict(index=i,theta=h['mano_pose']['thetas'],wrist=h['mano_pose']['wrist_xform'],beta=beta,R=pose[:3,:3],T=pose[:3,3]))
    indices=[r['index'] for r in records];assert len(indices)==len(set(indices)) and len(records)>50
    beta=torch.tensor(np.asarray([r['beta'] for r in records]),dtype=torch.float32,device='cuda')
    theta=torch.tensor(np.asarray([r['theta'] for r in records]),dtype=torch.float32,device='cuda')
    root=torch.tensor(np.asarray([r['wrist'] for r in records]),dtype=torch.float32,device='cuda')
    layer=smplx.create(str(asset),'mano',use_pca=True,is_rhand=True,num_pca_comps=15).cuda().eval();codec=NativeMANO()
    with torch.no_grad():
        gt=layer(betas=beta,global_orient=root[:,:3],hand_pose=theta,transl=root[:,3:],return_verts=True,return_full_pose=True)
        globalR=exp_so3(gt.full_pose[:,:3]);local=exp_so3(gt.full_pose[:,3:].reshape(-1,15,3))
        xyz,vertices=codec.decode(gt.joints[:,0],globalR,local,beta,torch.ones(len(indices),device='cuda',dtype=torch.bool),vertices=True)
        mesh_error=float((vertices-gt.vertices).abs().max());assert mesh_error<1e-6
        camR=torch.tensor(np.asarray([r['R'] for r in records]),dtype=torch.float32,device='cuda');camT=torch.tensor(np.asarray([r['T'] for r in records]),dtype=torch.float32,device='cuda')
        xyz_camera=torch.einsum('nji,nkj->nki',camR,xyz-camT[:,None]);root_camera=camR.transpose(1,2)@globalR
        fids=raw['feature_ids'][indices,8];base=raw['state'][fids].cuda();pred_beta=raw['betas'][fids].cuda()
        deltaR=torch.cat([(exp_so3(base[:,3:6]).transpose(-1,-2)@root_camera)[:,None],exp_so3(base[:,6:].reshape(-1,15,3)).transpose(-1,-2)@local],1)
        rotvec=torch.tensor(Rotation.from_matrix(deltaR.cpu().numpy().reshape(-1,3,3)).as_rotvec().reshape(-1,16,3),device='cuda',dtype=torch.float32)
        normalized=rotvec/codec.scales[1:];delta=torch.cat([torch.zeros(len(indices),1,3,device='cuda'),normalized],1).clamp(-1,1)
        reconstructed=codec.decode(xyz_camera[:,0],root_camera,local,beta,torch.ones(len(indices),device='cuda',dtype=torch.bool))
        coordinate_error=float((reconstructed-xyz_camera).abs().max());assert coordinate_error<2e-6
        oracle_predshape=codec.decode(xyz_camera[:,0],root_camera,local,pred_beta,torch.ones(len(indices),device='cuda',dtype=torch.bool))
        centered=lambda x:x-x[:,:1]
        shape_floor=float((centered(oracle_predshape)-centered(xyz_camera)).norm(dim=-1)[:,1:].mean()*1000)
    target=dict(indices=torch.tensor(indices),delta=delta.cpu(),gt=xyz_camera.cpu(),valid=torch.ones(len(indices),21,dtype=torch.bool),
        root_rotation=root_camera.cpu(),local_rotation=local.cpu(),beta_GT=beta.cpu())
    torch.save(target,R/'data/native_targets.pt')
    report=dict(complete=True,counts=dict(counts),roles=dict(collections.Counter(rows[i]['role'] for i in indices)),subjects=dict(collections.Counter(rows[i]['subject'] for i in indices)),
        native_targets_sha256=sha(R/'data/native_targets.pt'),official_mesh_parity_max_m=mesh_error,camera_FK_parity_max_m=coordinate_error,
        residual_components_saturated=int((normalized.abs()>1).sum()),residual_components_total=normalized.numel(),predicted_shape_only_relative_floor_mm=shape_floor,
        parameter_target='SO3 log(coarse^-1 * actual native GT), scales0.2rad root/0.8rad local; saturation explicitly counted',
        point_target='native21 WiLoR exact joint and fingertip convention decoded at GT shape; exclude zero wrist for relative metric',
        no_UmeTrack_fitting=True,no_labels_from_dev_calibrate_or_validation=True)
    (R/'data/ready.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
