"""Native21 audit of sealed full-chain replay; right hand only."""
import hashlib,json,sys,tarfile
from pathlib import Path
import numpy as np,torch,smplx
from scipy.optimize import linear_sum_assignment
R=Path('/mnt/why/native_mano_pool_v7_20261019');V2=Path('/mnt/why/native_mano_dit_v2_20261019')
sys.path.insert(0,str(V2/'src'));sys.path.insert(0,'/mnt/why/native_mano_multiseq_v3_20261019/src');sys.path.insert(0,'/mnt/why/public_headhand_h20_20261019/src')
from native_mano import NativeMANO,common,exp_so3
from geometry import transform
from evaluate_validation import iou,stats,protection
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(4);assert json.loads((R/'receipts/full_pipeline_done.json').read_text())['complete']
    view=R/'full_pipeline';rows=json.loads((view/'outputs/wilor_clip2102/observations.json').read_text())['frames']
    files=dict(BA=view/'outputs/mano_baseline_clip2102/parameters.npz',old_v5_guard=Path('/mnt/why/native_mano_supervision_v5_20261019/full_pipeline/outputs/ba_rollout_dit_guarded/parameters.npz'),small=view/'outputs/ba_small/parameters.npz',small_guarded=view/'outputs/ba_small_guarded/parameters.npz',full=view/'outputs/ba_full/parameters.npz',full_guarded=view/'outputs/ba_full_guarded/parameters.npz')
    data={k:np.load(p) for k,p in files.items()}
    for p in files.values():assert (p.parent/'seal.json').exists()
    beta=[];theta=[];root=[];rot=[];trans=[];ids=[]
    with tarfile.open('/mnt/why/public_headhand_h20_20261019/assets/clip-002102-full.tar') as tar:
        shape=json.load(tar.extractfile('__hand_shapes.json__'))['mano']
        for fi in range(150):
            h=json.load(tar.extractfile(f'{fi:06}.hands.json'));cal=json.load(tar.extractfile(f'{fi:06}.cameras.json'))['214-1'];pose=transform(cal['T_world_from_camera']);targets=[]
            for side,x in h.items():
                if '214-1' not in x.get('boxes_amodal',{}):continue
                box=np.array(x['boxes_amodal']['214-1']);box[[0,2]]=np.clip(box[[0,2]],0,1408);box[[1,3]]=np.clip(box[[1,3]],0,1408)
                if np.maximum(box[2:]-box[:2],0).prod()>=64:targets.append((side,box))
            obs=[i for i,r in enumerate(rows) if r['frame_index']==fi]
            overlap=np.array([[iou(np.array(rows[i]['box_xyxy']),t[1]) for t in targets] for i in obs]);a,b=linear_sum_assignment(1-overlap)
            for x,y in zip(a,b):
                if overlap[x,y]<.1 or targets[y][0]!='right' or not h['right'].get('mano_pose'):continue
                target=h['right']['mano_pose'];ids.append(obs[x]);beta.append(shape);theta.append(target['thetas']);root.append(target['wrist_xform']);rot.append(pose[:3,:3]);trans.append(pose[:3,3])
    layer=smplx.create(str(common.RUN/'assets/MANO_RIGHT.pkl'),'mano',use_pca=True,is_rhand=True,num_pca_comps=15).eval();codec=NativeMANO(device='cpu');tensor=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32);root=tensor(root);beta=tensor(beta)
    with torch.no_grad():
        out=layer(betas=beta,global_orient=root[:,:3],hand_pose=tensor(theta),transl=root[:,3:],return_verts=True,return_full_pose=True)
        gt=codec.decode(out.joints[:,0],exp_so3(out.full_pose[:,:3]),exp_so3(out.full_pose[:,3:].reshape(-1,15,3)),beta,torch.ones(len(ids),dtype=torch.bool)).numpy()
    gt=np.einsum('nji,nkj->nki',np.asarray(rot),gt-np.asarray(trans)[:,None]);pred={k:v['joints_camera'][ids] for k,v in data.items()};rel=lambda x:x-x[:,:1]
    errors={k:np.linalg.norm(p-gt,axis=-1)*1000 for k,p in pred.items()};relative={k:np.linalg.norm(rel(p)-rel(gt),axis=-1)[:,1:]*1000 for k,p in pred.items()}
    root_diagnostics={}
    for name,p in pred.items():
        delta=p[:,0]-gt[:,0]
        variants={
            'baseline_BA_wrist_rigid_translation_only':p+(pred['BA'][:,0]-p[:,0])[:,None],
            'oracle_GT_wrist_diagnostic_only':p+(gt[:,0]-p[:,0])[:,None],
        }
        for axis,label in [(2,'oracle_GT_wrist_Z_diagnostic_only'),((0,1),'oracle_GT_wrist_XY_diagnostic_only')]:
            change=np.zeros_like(delta);change[:,axis]=-delta[:,axis];variants[label]=p+change[:,None]
        root_diagnostics[name]=dict(signed_bias_XYZ_mm=(delta.mean(0)*1000).tolist(),wrist_abs_Z_mm=stats(np.abs(delta[:,2])*1000),wrist_XY_norm_mm=stats(np.linalg.norm(delta[:,:2],axis=1)*1000),
            camera21_after_root_replacement_mm={k:stats((np.linalg.norm(v-gt,axis=-1)*1000).ravel()) for k,v in variants.items()})
    report=dict(root_diagnostics=root_diagnostics,oracle_scope='Posthoc error decomposition only. Oracle GT replacements are NOT predictions or inference inputs. Baseline-wrist replacement is a diagnostic rigid root transform, not a writer or accepted new algorithm.',complete=True,hand_frames=len(ids),side='right only',native_GT='dataset MANO parameters decoded with existing RIGHT asset and WiLoR21 semantics; annotation-original asset version not independently verified',historical_diagnostic_replay=True,
        frozen_inputs_sha256={k:sha(p) for k,p in files.items()},metrics={k:dict(camera21_mm=stats(e.ravel()),native_relative20_mm=stats(relative[k].ravel()),wrist_mm=stats(e[:,0]),preservation=protection(relative['BA'],relative[k],np.ones_like(relative[k],bool))) for k,e in errors.items()},production_admitted=False)
    (R/'receipts/root_error_diagnostic.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
