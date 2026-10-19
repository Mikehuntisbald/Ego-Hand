"""Matched spatial/pool image heads, full optimizer/RNG and actual resume probes."""
import argparse,copy,hashlib,json,math,os,random,sys,time
from pathlib import Path
import numpy as np,torch
from rgb_head import RGBObservation,loss,pixel_error,blend
R=Path('/mnt/why/native_rgb_observation_v9_20261019');V7=Path('/mnt/why/native_mano_pool_v7_20261019')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def checkpoint(p,x):
    temporary=p.with_suffix('.partial.pt');torch.save(x,temporary);temporary.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['spatial','pooled'],required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=600);ap.add_argument('--stop-after',type=int);ap.add_argument('--save-every',type=int,default=100);ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3';torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False
    random.seed(190019);np.random.seed(190019);torch.manual_seed(190019);torch.cuda.manual_seed_all(190019);a.output.mkdir(parents=True,exist_ok=True)
    assert json.loads((R/'data/ready.json').read_text())['complete'];labels=torch.load(R/'data/targets.pt',map_location='cpu',weights_only=False);raw=torch.load(V7/'data/inputs.pt',map_location='cpu',weights_only=False)
    bank=np.load(V7/'data/rgb.npy',mmap_mode='r');rgb=torch.from_numpy(np.array(bank[labels['feature_ids'].numpy()])).cuda();target={k:v.cuda() for k,v in labels.items() if torch.is_tensor(v)};rows=[raw['rows'][int(i)] for i in labels['indices']]
    train=[i for i,r in enumerate(rows) if r['role']=='train'];select=[i for i,r in enumerate(rows) if r['role']=='dev_select'];assert len(train)==1330 and len(select)==25
    people=sorted({rows[i]['subject'] for i in train});by_person={p:[i for i in train if rows[i]['subject']==p] for p in people}
    model=RGBObservation(mode=a.mode).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);batch_rng=random.Random(910019);start=0;best_score=float('inf');best_model=None;history=[];batch_plan=[]
    if a.resume:
        saved=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False);assert saved['mode']==a.mode and saved['total_steps']==a.steps;model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);start=saved['step'];best_score=saved['best_score'];best_model=saved['best_model'];history=saved['history'];batch_plan=saved['batch_plan']
        torch.set_rng_state(saved['rng_cpu']);torch.cuda.set_rng_state_all(saved['rng_cuda']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng']);batch_rng.setstate(saved['batch_rng'])
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};gradient={};began=time.time()
    def predict(ids):
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):return model(rgb[ids],target['coarse_uv'][ids])
    def selection():
        out=predict(select);gt=target['uv'][select];mask=target['inband'][select];e=pixel_error(out['uv'],gt);base=pixel_error(target['coarse_uv'][select],gt);good=(base<=5)&mask;harm=int((good&(e>base+1)).sum());score=float(e[mask].mean())+20*harm/max(1,int(good.sum()));bu,w=blend(out,target['coarse_uv'][select]);be=pixel_error(bu,gt)
        return dict(score=score,inband_crop_error_px=float(e[mask].mean()),baseline_px=float(base[mask].mean()),blended_px=float(be[mask].mean()),accurate_points=int(good.sum()),head_harm_over1px=harm,blend_harm_over1px=int((good&(be>base+1)).sum()),blend_weight_mean=float(w.mean()))
    for step in range(start+1,a.steps+1):
        selected=batch_rng.sample(people,3);ids=[batch_rng.choice(by_person[p]) for p in selected for _ in range(4)];batch_plan.append(ids)
        rate=2e-5+(2e-4-2e-5)*.5*(1+math.cos(math.pi*(step-1)/600))
        for group in optimizer.param_groups:group['lr']=rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(rgb[ids],target['coarse_uv'][ids]);total,parts=loss(model,out,{k:v[ids] for k,v in target.items()})
        assert torch.isfinite(total);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        if step==start+1:
            gradient={k:float(v.grad.detach().abs().sum()) for k,v in model.named_parameters() if v.grad is not None};assert gradient and all(np.isfinite(list(gradient.values())))
        if step%100==0 or step==a.steps:
            metrics=selection();record=dict(step=step,loss=float(total.detach()),**parts,selection=metrics,elapsed_s=time.time()-began);history.append(record);print(json.dumps(record),flush=True)
            if metrics['score']<best_score:best_score=metrics['score'];best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};checkpoint(a.output/'best_development.pt',dict(model=best_model,mode=a.mode,width=128,step=step,selection=metrics,GT_inference=False,production_admitted=False))
            (a.output/'progress.json').write_text(json.dumps(record,indent=2))
        if step%a.save_every==0 or step==a.steps or a.stop_after==step:
            saved=dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),batch_rng=batch_rng.getstate(),step=step,mode=a.mode,total_steps=a.steps,best_score=best_score,best_model=best_model,history=history,batch_plan=batch_plan)
            checkpoint(a.output/'last.pt',saved)
        if a.stop_after==step:return
    changes={k:float((v.detach().cpu()-initial[k]).abs().sum()) for k,v in model.state_dict().items() if k!='grid'}
    done=dict(complete=True,steps=a.steps,mode=a.mode,gradient_sums=gradient,actual_weight_updates=changes,batch_plan_sha256=hashlib.sha256(json.dumps(batch_plan).encode()).hexdigest(),train_examples=len(batch_plan)*12,visual_backbone_frozen=True,GT_inference=False,production_admitted=False)
    (a.output/'done.json').write_text(json.dumps(done,indent=2));print(json.dumps({k:v for k,v in done.items() if k not in ['gradient_sums','actual_weight_updates']}),flush=True)
if __name__=='__main__':main()
