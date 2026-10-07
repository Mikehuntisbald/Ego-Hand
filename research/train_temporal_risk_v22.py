"""Subject-excluded historical-point error heads, not visibility estimation."""
import argparse, json, time
import numpy as np
import torch
from torch.nn import functional as F
from hand3d_v8_common import V7, save
from hand3d_data_v7 import batch
from hand3d_risk_v7 import Risk3D
from temporal_risk_features_v22 import features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device',default='cuda:0')
    args = ap.parse_args()
    torch.set_num_threads(4)
    root = V7.parent/'temporal_reliability_v22'
    root.mkdir(exist_ok=True)
    assert not (root/'done.json').exists()
    source = V7.parent/'context_data_v17/control'
    raw = torch.load(source/'dense_data.pt',weights_only=False,mmap=True)
    data = {k:v.to(args.device) if torch.is_tensor(v) else v for k,v in raw.items()}
    extra = {k:v.to(args.device) for k,v in torch.load(source/'trajectory_labels.pt',weights_only=False,mmap=True).items()}
    N = len(data['roles'])
    start = time.time()
    with torch.no_grad():
        pieces = [features(data,torch.arange(i,min(i+32,N),device=args.device)) for i in range(0,N,32)]
        x = torch.cat(pieces)
        del pieces
        # Norm errors are invariant to the rigid current-eye alignment.
        xyz = torch.cat([batch(data,torch.arange(i,min(i+64,N),device=args.device))['xyz'] for i in range(0,N,64)])
        available = data['available_bank'][data['feature_ids']] & (data['feature_ids']>0)[...,None]
        valid = extra['valid'] & available
        gt = extra['gt']
        camera = (xyz-gt).norm(dim=-1)
        relative = ((xyz-xyz[:,:,5:6])-(gt-gt[:,:,5:6])).norm(dim=-1)
        y = torch.stack([camera>.02,relative>.02],-1).float()
        mask = valid[...,None].expand(-1,-1,-1,2)
        probe_ids = torch.arange(2,device=args.device)
        clean = features(data,probe_ids)
        poisoned = dict(data,gt=data['gt']+9876,valid=~data['valid'])
        assert torch.equal(clean,features(poisoned,probe_ids))
    save(root/'feature_preflight.json',dict(passed=True,shape=list(x.shape),gt_poison_exact=True,
        target='Perframe perjoint camera/relative prediction error>20mm; finite3D GT and observable inputs only; not visibility labels',
        masks='No absentframe teacher supervision; trajectory labels track same centerhand',seconds=time.time()-start))
    roles,subjects = np.array(data['roles']),np.array(data['subjects'])
    train = np.where(roles=='train')[0]
    dev = torch.tensor(np.where(roles=='dev_select')[0],device=args.device)
    cal = torch.tensor(np.where(roles=='dev_calibrate')[0],device=args.device)
    folds = json.loads((V7/'risk_calibration.json').read_text())['folds']
    assert set(subjects[train])==set(folds)
    oof = torch.zeros(N,17,20,2,device=args.device)
    lineage = []
    for excluded in [None,0,1,2]:
        name = 'all' if excluded is None else f'fold{excluded}'
        ix = torch.tensor([i for i in train if excluded is None or folds[subjects[i]]!=excluded],device=args.device)
        held = torch.tensor([i for i in train if excluded is not None and folds[subjects[i]]==excluded],device=args.device)
        trained_subjects = sorted({subjects[i] for i in ix.cpu().tolist()})
        excluded_subjects = sorted({subjects[i] for i in held.cpu().tolist()})
        assert not set(trained_subjects)&set(excluded_subjects)
        torch.manual_seed(202610114)
        model = Risk3D(172).to(args.device)
        observed = x[ix][valid[ix]]
        model.mean.copy_(observed.mean(0))
        model.scale.copy_(observed.std(0).clamp_min(.02))
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0005,weight_decay=.03)
        best = float('inf')
        for step in range(1,1601):
            model.train()
            picked = ix[torch.randint(len(ix),(32,),device=args.device)]
            loss = F.binary_cross_entropy_with_logits(model(x[picked])[mask[picked]],y[picked][mask[picked]])
            assert torch.isfinite(loss)
            optimizer.zero_grad(set_to_none=True)
            loss.backward();optimizer.step()
            if step%200==0:
                model.eval()
                with torch.no_grad():
                    score = float(F.binary_cross_entropy_with_logits(model(x[dev])[mask[dev]],y[dev][mask[dev]]))
                if score<best:
                    best=score
                    torch.save(dict(model=model.state_dict(),dim=172,step=step,folds=folds,
                        trained_subjects=trained_subjects,excluded_subjects=excluded_subjects),root/f'risk_{name}.pt')
        checkpoint = torch.load(root/f'risk_{name}.pt',weights_only=False,map_location=args.device)
        model.load_state_dict(checkpoint['model']);model.eval()
        with torch.no_grad():
            logits = torch.cat([model(x[i:i+64]) for i in range(0,N,64)])
        if excluded is None:
            all_logits = logits
        else:
            oof[held] = logits[held]
        lineage.append(dict(name=name,step=checkpoint['step'],dev_bce=best,
            trained_subjects=trained_subjects,excluded_subjects=excluded_subjects))
        print(json.dumps(dict(**lineage[-1],seconds=time.time()-start)),flush=True)
    temperatures=[]
    for channel in range(2):
        values=[(float(F.binary_cross_entropy_with_logits(all_logits[cal,:,:,channel][valid[cal]]/t,
            y[cal,:,:,channel][valid[cal]])),float(t)) for t in np.linspace(.5,3,51)]
        temperatures.append(min(values)[1])
    temperature=torch.tensor(temperatures,device=args.device)
    joint=(all_logits/temperature).sigmoid()
    train_probability=joint.clone()
    train_probability[train]=(oof[train]/temperature).sigmoid()
    torch.save(dict(joint=joint.cpu(),train_oof=train_probability.cpu()),root/'risk_probabilities.pt')
    save(root/'calibration.json',dict(temperature=temperatures,lineage=lineage,folds=folds,
        selection='Dev_select checkpoint, dev_calibrate temperature; no previousfresh or retainedfailures',
        scope='Own-subject riskhead exclusion only; sharedvisual/upstream pretraining notfullOOF. Probability ofXYZerror, not missingfinger visibility. Not yet recovery evidence.'))
    save(root/'done.json',dict(complete=True,windows=N,steps_per_head=1600,seconds=time.time()-start))
    print((root/'done.json').read_text(),flush=True)

if __name__=='__main__':
    main()
