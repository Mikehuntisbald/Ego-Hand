"""Re-fit GT-free error-risk inputs after changing real temporal observations."""
import argparse,json,time
import numpy as np,torch
from torch.nn import functional as F
from hand3d_v8_common import V7,save
from hand3d_data_v7 import batch,risk_features
from hand3d_risk_v7 import Risk3D
ROOT=V7.parent/'aligned_density_v13'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--arm',choices=['sparse','dense'],required=True);ap.add_argument('--device',default='cuda:0');args=ap.parse_args();torch.set_num_threads(4)
    assert json.loads((ROOT/'ready.json').read_text())['complete'];run=ROOT/f'risk_{args.arm}';run.mkdir(exist_ok=True);assert not (run/'done.json').exists()
    raw=torch.load(ROOT/f'{args.arm}_data.pt',weights_only=False,mmap=True);data={k:v.to(args.device) if torch.is_tensor(v) else v for k,v in raw.items()};N=len(data['roles']);parts=[]
    with torch.no_grad():
        for start in range(0,N,64):parts.append(risk_features(batch(data,torch.arange(start,min(start+64,N),device=args.device))))
    x=torch.cat(parts);base=data['xyz_camera_bank'][data['feature_ids'][:,8]];gt=data['gt'];valid=data['valid'];roles=np.array(data['roles']);subjects=np.array(data['subjects']);camera=(base-gt).norm(dim=-1);relative=((base-base[:,5:6])-(gt-gt[:,5:6])).norm(dim=-1);y=torch.stack([camera>.02,relative>.02],-1).float();mask=valid[...,None].expand(-1,-1,2)
    train=np.where(roles=='train')[0];dev=torch.tensor(np.where(roles=='dev_select')[0],device=x.device);cal=torch.tensor(np.where(roles=='dev_calibrate')[0],device=x.device)
    fold=json.loads((V7/'risk_calibration.json').read_text())['folds'];assert set(subjects[train])==set(fold);oof=torch.zeros(N,20,2,device=x.device);lineage=[];started=time.time()
    for excluded in [None,0,1,2]:
        name='all' if excluded is None else f'fold{excluded}';ix=torch.tensor([j for j in train if excluded is None or fold[subjects[j]]!=excluded],device=x.device);held=torch.tensor([j for j in train if excluded is not None and fold[subjects[j]]==excluded],device=x.device)
        excluded_subjects=sorted({subjects[j] for j in held.cpu().tolist()});trained_subjects=sorted({subjects[j] for j in ix.cpu().tolist()});assert not set(excluded_subjects)&set(trained_subjects)
        torch.manual_seed(202610114);model=Risk3D(x.shape[-1]).to(x.device);model.mean.copy_(x[ix].mean((0,1)));model.scale.copy_(x[ix].std((0,1)).clamp_min(.02));opt=torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.03);best=float('inf')
        for step in range(1,1601):
            model.train();i=ix[torch.randint(len(ix),(128,),device=x.device)];loss=F.binary_cross_entropy_with_logits(model(x[i])[mask[i]],y[i][mask[i]]);assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            if step%200==0:
                model.eval()
                with torch.no_grad():score=float(F.binary_cross_entropy_with_logits(model(x[dev])[mask[dev]],y[dev][mask[dev]]))
                if score<best:best=score;torch.save(dict(model=model.state_dict(),dim=x.shape[-1],step=step,folds=fold,excluded_subjects=excluded_subjects,trained_subjects=trained_subjects),run/f'risk_{name}.pt')
        ck=torch.load(run/f'risk_{name}.pt',weights_only=False,map_location=x.device);model.load_state_dict(ck['model']);model.eval()
        with torch.no_grad():logits=model(x)
        if excluded is None:all_logits=logits
        else:oof[held]=logits[held]
        lineage.append(dict(name=name,excluded_subjects=excluded_subjects,trained_subjects=trained_subjects,selected_step=ck['step'],dev_bce=best));print(json.dumps(dict(arm=args.arm,**lineage[-1],seconds=time.time()-started)),flush=True)
    temps=[]
    for channel in range(2):
        choices=[(float(F.binary_cross_entropy_with_logits(all_logits[cal,:,channel][valid[cal]]/temp,y[cal,:,channel][valid[cal]])),float(temp)) for temp in np.linspace(.5,3,51)];temps.append(min(choices)[1])
    temperature=torch.tensor(temps,device=x.device);joint=(all_logits/temperature).sigmoid();train_prob=joint.clone();train_prob[train]=(oof[train]/temperature).sigmoid();torch.save(dict(joint=joint.cpu(),train_oof=train_prob.cpu()),run/'risk_probabilities.pt')
    save(run/'calibration.json',dict(temperature=temps,folds=fold,lineage=lineage,outputs=['P(camera error >20mm)','P(wrist-relative error >20mm)'],selection='Train folds exclude own subjects; dev_select checkpoint and dev_calibrate temperature; no test/fresh rows',scope='Only risk head excluded; shared visual training/pretraining overlap not excluded',arm=args.arm));save(run/'done.json',dict(complete=True,seconds=time.time()-started,windows=N))

if __name__=='__main__':main()
