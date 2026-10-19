import argparse,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np,torch
from training_data import TrainingData
from risk_model import NativeQuality,objective
R=Path('/mnt/why/native_candidate_risk_v11_20261019')
ARMS=['pose_base','RGB_base','RGB_preserve']
def save(p,x):
    temp=p.with_suffix('.partial.pt');torch.save(x,temp);temp.replace(p)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=ARMS,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--steps',type=int,default=600);ap.add_argument('--stop-after',type=int);ap.add_argument('--save-every',type=int,default=100);ap.add_argument('--resume',action='store_true');a=ap.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')=='3'
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);random.seed(190019);np.random.seed(190019);torch.manual_seed(190019);torch.cuda.manual_seed_all(190019);data=TrainingData();model=NativeQuality(use_rgb=a.arm!='pose_base').cuda();opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.01);rng=random.Random(910019);a.output.mkdir(parents=True,exist_ok=True)
    by_person={}
    for j in data.train:by_person.setdefault(data.rows[data.indices[j]]['subject'],[]).append(j)
    people=sorted(by_person);assert len(people)==6;step0=0;best=float('inf');best_model=None;history=[];plan=[]
    if a.resume:
        ck=torch.load(a.output/'last.pt',map_location='cpu',weights_only=False);assert ck['arm']==a.arm and ck['total_steps']==a.steps;model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);step0=ck['step'];best=ck['best_score'];best_model=ck['best_model'];history=ck['history'];plan=ck['batch_plan'];torch.set_rng_state(ck['rng_cpu']);torch.cuda.set_rng_state_all(ck['rng_cuda']);random.setstate(ck['python_rng']);np.random.set_state(ck['numpy_rng']);rng.setstate(ck['batch_rng'])
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};grads={};start=time.time()
    def select():
        bs,cs,ys=data.get(data.select)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(bs,cs)
        choice=out['score'].argmin(1);idx=torch.arange(len(choice),device='cuda');error=ys['error_mm'][idx,choice];reference=ys['error_mm'][:,0];good=reference<=10;harm=int((good&(error>reference+1)).sum());hard=reference>20;metric=float(error.mean())+20*harm/max(1,int(good.sum()));prob=out['harm_probability'];labels=ys['harm_on_good'].float()
        return dict(score=metric,native_relative20_mm=float(error.mean()),reference_mm=float(reference.mean()),accurate_points=int(good.sum()),harmed_over1mm=harm,hard_points=int(hard.sum()),recovered_to20=int((hard&(error<=20)).sum()),candidate_counts=torch.bincount(choice,minlength=6).cpu().tolist(),delta_MAE_nonreference_mm=float((out['delta_error_mm'][:,1:]-ys['delta_error_mm'][:,1:]).abs().mean()),harm_Brier_nonreference=float(((prob[:,1:]-labels[:,1:])**2).mean()))
    for step in range(step0+1,a.steps+1):
        chosen=rng.sample(people,3);ids=[rng.choice(by_person[p]) for p in chosen for _ in range(3)];plan.append(ids);b,c,y=data.get(ids);rate=2e-5+(2e-4-2e-5)*.5*(1+math.cos(math.pi*(step-1)/600))
        for group in opt.param_groups:group['lr']=rate
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=model(b,c);total,parts=objective(out,y,extra_preservation=a.arm=='RGB_preserve')
        assert torch.isfinite(total);total.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        if step==step0+1:grads={k:float(v.grad.abs().sum()) for k,v in model.named_parameters() if v.grad is not None}
        if step%100==0 or step==a.steps:
            metrics=select();row=dict(step=step,loss=float(total.detach()),selection=metrics,seconds=time.time()-start,**parts);history.append(row);(a.output/'progress.json').write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
            if metrics['score']<best:
                best=metrics['score'];best_model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};save(a.output/'best_development.pt',dict(model=best_model,arm=a.arm,width=96,use_rgb=a.arm!='pose_base',selected_step=step,selection=metrics,GT_inference=False,production_admitted=False))
        if step%a.save_every==0 or step==a.steps or step==a.stop_after:
            save(a.output/'last.pt',dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=opt.state_dict(),rng_cpu=torch.get_rng_state(),rng_cuda=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),batch_rng=rng.getstate(),step=step,arm=a.arm,total_steps=a.steps,best_score=best,best_model=best_model,history=history,batch_plan=plan))
        if step==a.stop_after:return
    updates={k:float((v.detach().cpu()-initial[k]).abs().sum()) for k,v in model.state_dict().items()};done=dict(complete=True,arm=a.arm,steps=a.steps,frame_examples=len(plan)*9,batch_plan_sha256=hashlib.sha256(json.dumps(plan).encode()).hexdigest(),gradient_sums=grads,actual_updates=updates,GT_input=False,visual_backbone_and_DiT_frozen=True,production_admitted=False);(a.output/'done.json').write_text(json.dumps(done,indent=2));print(json.dumps({k:v for k,v in done.items() if k not in ['gradient_sums','actual_updates']}),flush=True)
if __name__=='__main__':main()
