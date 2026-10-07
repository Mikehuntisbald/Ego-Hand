"""Label-fitted calibration parameters are frozen constants at inference."""
import numpy as np,torch
from metrics_3d import EVAL_INDICES,compare

def raw_sigma(confidence):
    scale=torch.tensor([.05]+[.025]*20,dtype=confidence.dtype,device=confidence.device)
    return -confidence.clamp(1e-8,1-1e-8).log()*scale

def fit_uncertainty(coarse,gt,confidence,subjects):
    sigma=raw_sigma(confidence).clamp_min(.001)
    root=(coarse[:,5]-gt[:,5]).square().mean(-1,keepdim=True)
    rel=((coarse-coarse[:,5:6])-(gt-gt[:,5:6])).square().mean(-1)
    mse=torch.cat([root,rel],1);ratios=[]
    for subject in sorted(set(subjects)):
        mask=torch.tensor([s==subject for s in subjects]);ratios.append((mse[mask]/sigma[mask].square()).mean(0))
    factors=torch.stack(ratios).mean(0).sqrt().clamp(.5,8.)
    factors[6]=1. # redundant zero wrist-relative coordinate
    return dict(factors=factors.tolist(),training_subjects=sorted(set(subjects)),
        samples=len(coarse),fit='Equal-subject Gaussian variance scaling; denoiser sequences only',
        before_sigma_relative_mm=float(sigma[:,1:][:,EVAL_INDICES].mean()*1000),
        after_sigma_relative_mm=float((sigma*factors)[...,1:][:,EVAL_INDICES].mean()*1000),
        actual_relative_axis_rms_mm=float(rel[:,EVAL_INDICES].mean().sqrt()*1000))

def calibrated_confidence(confidence,fit):
    factors=torch.tensor(fit['factors'],device=confidence.device,dtype=confidence.dtype)
    return confidence.clamp(1e-8,1-1e-8).pow(factors).clamp(1e-6,1-1e-6)

def active_data(raw,method,fit=None):
    data=dict(raw)
    if method=='in_subject_dit':data['coarse']=raw['in_subject_coarse'];data['confidence']=raw['in_subject_confidence']
    if fit is not None:data['confidence']=calibrated_confidence(data['confidence'],fit)
    return data

def fit_probability(logits,proposal,coarse,gt,model):
    with torch.no_grad():
        full=coarse+model.unpack(proposal)
        helpful=((full-gt).norm(dim=-1)+.001<(coarse-gt).norm(dim=-1)).float()
        target=torch.cat([helpful[:,5:6],helpful],1)
        mask=model.mask.squeeze(-1).expand(len(gt),-1).bool()
        x=logits[mask].detach().float().cpu();y=target[mask].cpu()
    log_a=torch.tensor(0.,requires_grad=True);b=torch.tensor(0.,requires_grad=True)
    opt=torch.optim.Adam([log_a,b],lr=.03)
    for _ in range(400):
        opt.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(torch.nn.functional.softplus(log_a)*x+b,y)
        loss.backward();opt.step()
    a=float(torch.nn.functional.softplus(log_a).detach());bias=float(b.detach())
    before=x.sigmoid();after=(a*x+bias).sigmoid()
    def scores(p):
        ece=0.
        for low in np.linspace(0,.9,10):
            m=(p>=low)&(p<low+.1)
            if m.any():ece+=float(m.float().mean()*abs(p[m].mean()-y[m].mean()))
        return dict(brier=float(((p-y)**2).mean()),ece10=ece)
    return dict(a=a,b=bias,beneficial_rate=float(y.mean()),before=scores(before),after=scores(after),
        note='Monotonic Platt fit on held calibration sequence; targets score full generated proposal, not privileged inference inputs')

def wilson_upper(harmed,total):
    if not total:return 1.
    z=1.96;p=harmed/total;denom=1+z*z/total
    return (p+z*z/(2*total)+z*np.sqrt(p*(1-p)/total+z*z/(4*total*total)))/denom

def calibration_grid(model,coarse,gt,proposal,logits,probability,confidence):
    gates=(probability['a']*logits+probability['b']).sigmoid()*model.mask.squeeze(-1)
    c=coarse.numpy();g=gt.numpy();idx=EVAL_INDICES
    before=np.linalg.norm(c-g,axis=-1)*1000
    before_rel=np.linalg.norm((c-c[:,5:6])-(g-g[:,5:6]),axis=-1)*1000
    correct=before[:,idx]<=10;correct_rel=before_rel[:,idx]<=10
    low=confidence[:,1:].numpy()<float(torch.quantile(confidence[:,1:][:,idx].reshape(-1),.2));low[:,5]=False
    base=compare(c,c,g);best=None;grid=[]
    for threshold in [0,.25,.5,.65,.75,.85,.9,.95,.99]:
        selected=gates*(gates>=threshold)
        for strength in [0,.025,.05,.1,.2,.35,.5,.75,1.]:
            pred=model.apply_gates(proposal,coarse,selected*strength).numpy()
            after=np.linalg.norm(pred-g,axis=-1)*1000
            after_rel=np.linalg.norm((pred-pred[:,5:6])-(g-g[:,5:6]),axis=-1)*1000
            harms=int(((after[:,idx]>before[:,idx]+1)&correct).sum())
            relharms=int(((after_rel[:,idx]>before_rel[:,idx]+1)&correct_rel).sum())
            camera_harm=harms/max(1,correct.sum());rel_harm=relharms/max(1,correct_rel.sum())
            cu=wilson_upper(harms,int(correct.sum()));ru=wilson_upper(relharms,int(correct_rel.sum()))
            camera=float(after[:,idx].mean());relative=float(after_rel[:,idx].mean())
            admissible=(camera_harm<=.025 and rel_harm<=.025 and cu<=.05 and ru<=.05 and camera<=base['coarse']['mpjpe19_mm']+.1)
            if strength==0:admissible=True # exact zero update fallback
            low_error=float(after_rel[low].mean());score=camera+relative+.5*low_error
            row=dict(threshold=threshold,strength=strength,score=score,camera_mm=camera,relative_mm=relative,
                low_confidence_relative_mm=low_error,camera_harm=camera_harm,relative_harm=rel_harm,camera_harm_wilson_upper=cu,relative_harm_wilson_upper=ru,admissible=bool(admissible))
            grid.append(row)
            if admissible and (best is None or score<best['score']):best=row
    return dict(**best,probability=probability),grid

def frozen_gates(model,logits,operating):
    p=operating['probability'];g=(p['a']*logits+p['b']).sigmoid()*model.mask.squeeze(-1)
    return g*(g>=operating['threshold'])*operating['strength']
