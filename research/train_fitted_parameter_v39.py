"""Train RGB temporal parameters; checkpoint only on fixed dev_select."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from hand3d_v8_common import V7,save,metrics,score
from parameter_codec_v31 import OUT,observation_batch
from semantic_parameter_model_v36 import SemanticParameterHand as ParameterTemporalHand

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['regression','dit'],default='regression');ap.add_argument('--device',default='cuda:0');ap.add_argument('--steps',type=int,default=4000);ap.add_argument('--batch',type=int,default=8);a=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(202610131);assert json.loads((OUT/'refined_target_representability.json').read_text())['passed']
    run=V7.parent/'fitted_parameter_v39'/a.kind;run.mkdir(parents=True,exist_ok=True);assert not (run/'done.json').exists()
    data=torch.load(V7.parent/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True)
    data={k:v.to(a.device) if torch.is_tensor(v) else v for k,v in data.items()}
    target={k:v.to(a.device) for k,v in torch.load(OUT/'refined_targets.pt',weights_only=False,mmap=True).items()}
    coarse=torch.load(V7.parent/'observation_ik_v38/coarse_state_bank.pt',weights_only=False,mmap=True).to(a.device);right=torch.load(OUT/'predicted_right_bank.pt',weights_only=False).to(a.device)
    probabilities=torch.load(V7.parent/'side_data_v16/consensus/risk_dense/risk_probabilities.pt',weights_only=False)
    prob=probabilities['train_oof'].to(a.device);devprob=probabilities['joint'].to(a.device)
    bank=torch.load(V7.parent/'side_data_v16/consensus/native_bank.pt',weights_only=False,mmap=True).to(a.device)
    train=torch.tensor([i for i,r in enumerate(data['roles']) if r=='train'],device=a.device);dev=torch.tensor([i for i,r in enumerate(data['roles']) if r=='dev_select'],device=a.device)
    model=ParameterTemporalHand(a.kind,a.device).to(a.device);ck=torch.load(V7.parent/'side_native_v16/consensus/uniform_adaptive/best.pt',weights_only=False,map_location=a.device);transferred=model.load_visual_initial(ck['model'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01)
    config=dict(kind=a.kind,steps=a.steps,batch=a.batch,query_semantics='34scalarparameters linked to actual observation joints',coarse_seed='GT-free observationIK v38; own fitted root/shape retained; same objective/data/encoder asv36',seed=202610131,initial_v16_step=ck['step'],transferred_tensors=transferred,
        backbone='Frozenhandpretrained32layers; originalnative192x1280 preserved',trained='Visualprojection/localization, temporal4layer192/6heads, boundedparameter generator/sidehead',
        outputs='RootcameraXYZ,proper6Drotation,20boundedangles,5trainingPCAshape; FK generates17x20x3/current20x3meters',
        target='RefinedIK parameterlabels fromoriginalGT; GTshape/pose neverininput',policy='No postdecode pointXYZ clipping; automaticadoption requires fullaccuracy/protection/coherence validation',
        evaluation='Checkpointselection onlydev_select; dev_calibrate andfresh notread',natural_only=True)
    save(run/'config.json',config)
    def get(ids,p):
        b=observation_batch(data,ids,p,coarse,right,preserve_fitted=True);b['rgb_native']=bank[data['feature_ids'][ids]]
        return b
    @torch.inference_mode()
    def evaluate():
        model.eval();pred=[];states=[];sides=[]
        for start in range(0,len(dev),8):
            ix=dev[start:start+8];b=get(ix,devprob)
            with torch.autocast('cuda',dtype=torch.bfloat16):p=model.predict_parameters(b,seed=202610131+start)
            pred.append(p['xyz_camera_m']);states.append(p['state']);sides.append(p['right'])
        xyz=torch.cat(pred);m=metrics(xyz,data['original_base_for_evaluation'][dev],data['gt'][dev],data['valid'][dev]);key,ok=score(m)
        return key,m,dict(prediction=xyz.cpu(),state=torch.cat(states).cpu(),right=torch.cat(sides).cpu(),indices=dev.cpu())
    start=time.time();best=None;history=[]
    ix=train[:a.batch];b=get(ix,prob);tt={k:v[ix] for k,v in target.items()}
    with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.objective(b,tt,202610131)
    loss.backward();grad={k:float(p.grad.norm()) for k,p in model.named_parameters() if k in ['semantic_head.weight','rgb_project.weight','side_head.1.weight'] and p.grad is not None}
    assert all(np.isfinite(v) for v in grad.values()) and grad['semantic_head.weight']>0,grad
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        clean=model.predict_parameters(b);poison=dict(b,gt=torch.rand_like(tt['gt'])*999,gt_shape=torch.rand(len(ix),10,device=a.device),gt_right=1-tt['right'])
        polluted=model.predict_parameters(poison)
        assert torch.equal(clean['xyz_camera_m'],polluted['xyz_camera_m']) and torch.equal(clean['state'],polluted['state'])
    save(run/'preflight.json',dict(passed=True,gt_poison_exact=True,output_shape=list(clean['xyz_camera_m'].shape),gradient_norms=grad,loss=float(loss),parts=parts))
    for step in range(1,a.steps+1):
        model.train();ix=train[torch.randint(len(train),(a.batch,),device=a.device)];b=get(ix,prob);tt={k:v[ix] for k,v in target.items()}
        lr=.0002*min(step/100,1.)*(.2+.8*.5*(1+np.cos(np.pi*step/a.steps)))
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,parts=model.objective(b,tt,202610131+step)
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
        if step%100==0:print(json.dumps(dict(step=step,loss=float(loss),parts=parts,seconds=time.time()-start)),flush=True)
        if step%250==0:
            rank,m,p=evaluate();entry=dict(step=step,loss=float(loss),metrics=m,rank=list(rank),seconds=time.time()-start);history.append(entry);save(run/'history.json',history)
            if best is None or rank<best:
                best=rank;torch.save(dict(model=model.state_dict(),step=step,kind=a.kind,config=config),run/'best.pt');torch.save(p,run/'development_predictions.pt')
            print(json.dumps(entry),flush=True)
    ck=torch.load(run/'best.pt',weights_only=False,map_location='cpu');save(run/'done.json',dict(complete=True,steps=a.steps,selected_step=ck['step'],seconds=time.time()-start,default_changed=False))
    print((run/'done.json').read_text(),flush=True)

if __name__=='__main__':main()
