"""GT-free training proposal bank, frozen before native labels are opened."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np,torch
R=Path('/mnt/why/native_candidate_risk_v11_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019')
sys.path.insert(0,'/mnt/why/native_mano_dit_v2_20261019/src');sys.path.insert(0,'/mnt/why/native_mano_multiseq_v3_20261019/src')
from input_data import Inputs
from native_mano import NativeMANO
from native_dit import NativeMANODiT
from infer_validation import protected
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forbid(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=str(args[0]);assert not any(x in path for x in ['.hands.json','__hand_shapes.json__','native_targets.pt','quality_targets.pt','/targets.pt','training_targets.pt']), 'Hand-label access forbidden during input preparation'
def main():
    sys.addaudithook(forbid);assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3';assert int(subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())==0
    torch.set_num_threads(4);(R/'data').mkdir(parents=True,exist_ok=True);assert not (R/'data/candidates_ready.json').exists();data=Inputs();ids=[i for i,r in enumerate(data.rows) if r['role'] in ['train','dev_select'] and bool(data.raw['right'][data.raw['feature_ids'][i,8]])]
    ckpath=V7/'runs/full/best_development.pt';ck=torch.load(ckpath,map_location='cuda',weights_only=False);model=NativeMANODiT(**ck['config']).cuda().eval();model.load_state_dict(ck['model']);codec=NativeMANO();fixed={k:v.clone() for k,v in codec.state_dict().items()};chunks={};began=time.time()
    for role in ['train','dev_select']:
        eligible=[i for i in ids if data.rows[i]['role']==role];seed=410019 if role=='train' else 190019
        for begin in range(0,len(eligible),8):
            ix=eligible[begin:begin+8];b=data.batch(ix);joints=[];deltas=[]
            with torch.inference_mode():
                with torch.autocast('cuda',dtype=torch.bfloat16):draws=model.sample(b,draws=4,steps=10,seed=seed)
                raw=[draws.mean(0),torch.zeros_like(draws[0])]+list(draws)
                for d in raw:
                    result,alpha=protected(codec,b,d);native=codec.export_native(result);back=codec.decode_exported(native);assert float((back-result['joints']).abs().max())<1e-6;joints.append(result['joints'].cpu());deltas.append((d*alpha[:,None,None]).cpu())
            values=dict(indices=torch.tensor(ix),joints_camera21=torch.stack(joints,1),normalized_residual=torch.stack(deltas,1))
            for k,v in values.items():chunks.setdefault(k,[]).append(v)
            if begin%160==0:print(json.dumps(dict(stage=role,done=min(begin+8,len(eligible)),total=len(eligible),seconds=time.time()-began)),flush=True)
    assert all(torch.equal(v,fixed[k]) for k,v in codec.state_dict().items());pack={k:torch.cat(v) for k,v in chunks.items()};target=R/'data/candidates.pt';torch.save(pack,target)
    out=dict(complete=True,observations=len(pack['indices']),train_predicted_right=sum(data.rows[int(i)]['role']=='train' for i in pack['indices']),dev_select_predicted_right=sum(data.rows[int(i)]['role']=='dev_select' for i in pack['indices']),candidates_each=6,candidate_sha256=sha(target),source_inputs_sha256=sha(V7/'data/inputs.pt'),source_checkpoint_sha256=sha(ckpath),source_sha256=sha(Path(__file__)),hand_GT_read=False,fixed_FK_buffers=True,seconds=time.time()-began,production_admitted=False)
    (R/'data/candidates_ready.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
