"""Augmented-Lagrangian feasibility restoration; never relax acceptance caps.

The same parametric decoder and shared shape are used at every iteration.
XYZ vertices are never independently moved/clipped. Saved feasible tracks
still need accuracy/hard-case validation; feasibility is not recovery.
"""
import argparse, json, time
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_v8_common import V7, save
from joint_mano_model_v29 import violations, point_radii
from parameter_trajectory_model_v33 import ParameterTrajectoryDecoder
from evaluate_joint_kinematic_v30 import subset, identities, report, eligible, LIMITS


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1200);ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--name',default='joint_feasibility_v34')
    ap.add_argument('--closest-proposal',action='store_true')
    ap.add_argument('--data-weight',type=float,default=.1)
    ap.add_argument('--prior-run',default='parameter_trajectory_v33')
    a=ap.parse_args();torch.set_num_threads(4)
    rootpath=V7.parent;source=rootpath/'joint_mano_v28';priorpath=rootpath/a.prior_run;out=rootpath/a.name;out.mkdir(exist_ok=True)
    assert not (out/'selection.json').exists()
    allrows=json.loads((source/'rows.json').read_text());ids=torch.tensor([i for i,r in enumerate(allrows) if r['role']=='dev_select']);rows=[allrows[i] for i in ids]
    cache=subset(torch.load(source/'candidates.pt',weights_only=False,mmap=True),ids,len(allrows),a.device)
    obs={k:v.to(a.device) for k,v in torch.load(priorpath/'observations.pt',weights_only=False).items()}
    decoder=ParameterTrajectoryDecoder(a.device);initial=decoder.initialize(obs,cache,rows)
    previous=torch.load(priorpath/'whole_track.pt',weights_only=False)
    seed={k:v.to(a.device) for k,v in previous['parameters'].items()}
    group=initial['group'];groups=len(initial['keys']);device=a.device
    local=nn.Parameter(seed['local'].clone());rotation=nn.Parameter(seed['global_six'].clone());root=nn.Parameter(seed['root'].clone())
    shape=nn.Parameter(torch.atanh(seed['beta'].clamp(-3.999,3.999)/4))
    optimizer=torch.optim.Adam([dict(params=[local,rotation],lr=.004),dict(params=[root],lr=.00015),dict(params=[shape],lr=.003)])
    rc,rr,confirmed=point_radii(cache);base=cache['base']
    margin=.0005;motion_factor=.9;dual={};rho=20.
    accepted=torch.zeros(groups,device=device,dtype=torch.bool);output=cache['baseline'].clone();chosen={k:v.clone() for k,v in seed.items()}
    bestcost=torch.full((groups,),float('inf'),device=device)
    history=[];start=time.time()
    save(out/'protocol.json',dict(method='Augmented-Lagrangian parameter-space feasibility restoration',steps=a.steps,rho=rho,
        point_interior_margin_mm=margin*1000,motion_fit_fraction=motion_factor,acceptance='Original9.5/50mm radii and speed/acceleration limits unchanged',
        selection='Only dev_select, previously used data; no independent generalization claim',
        default_changed=False,feasibility_is_recovery=False,independent_point_clipping=False,
        feasible_selection='Closest observed v16 proposal' if a.closest_proposal else 'First feasible',data_weight=a.data_weight))
    def check_and_keep(world,params):
        xyz,point,bend,moves=violations(world,cache,initial,LIMITS)
        bad=(point>1e-6).any(-1)|(bend>1e-6).any(-1)|~torch.isfinite(xyz).all((-1,-2))
        if confirmed.any():bad|=((xyz!=base).any(-1)&confirmed).any(-1)
        count=torch.zeros(groups,device=device,dtype=torch.long);count.scatter_add_(0,group,bad.long())
        for name,values in moves.items():
            if values is None:continue
            excess=values.norm(dim=-1)>LIMITS[name]+1e-5
            while excess.ndim>1:excess=excess.any(-1)
            frame=initial['pair'][0] if 'speed' in name else initial['triple'][0]
            count.scatter_add_(0,group[frame],excess.long())
        cost=((xyz-cache['baseline'])/.03).square().mean((-1,-2))
        cost+=.75*(((xyz-xyz[:,5:6])-(cache['baseline']-cache['baseline'][:,5:6]))/.03).square().mean((-1,-2))
        total=torch.zeros(groups,device=device);total.scatter_add_(0,group,cost)
        total/=torch.bincount(group,minlength=groups).clamp_min(1)
        new=(count==0)&((total<bestcost) if a.closest_proposal else ~accepted)
        bestcost[new]=total[new];take=new[group]
        output[take]=xyz[take];accepted.logical_or_(new)
        for k in ['local','global_six','root']:chosen[k][take]=params[k][take]
        chosen['beta'][new]=params['beta'][new]
        return xyz,point,moves
    for step in range(a.steps+1):
        beta=4*shape.tanh();world=decoder(local,rotation,root,beta,group,initial['sign'])
        xyz,point,bend,moves=violations(world,cache,initial,LIMITS)
        dc=(xyz-base).norm(dim=-1);dr=((xyz-xyz[:,5:6])-(base-base[:,5:6])).norm(dim=-1)
        constraints={'camera':((dc-rc+margin)/.005).clamp_min(0),'relative':((dr-rr+margin)/.005).clamp_min(0),'bend':bend}
        for name,values in moves.items():
            if values is not None:constraints[name]=(values.norm(dim=-1)/(LIMITS[name]*motion_factor)-1).clamp_min(0)
        penalty=world.sum()*0
        for name,value in constraints.items():
            if name not in dual:dual[name]=torch.zeros_like(value)
            # Dual variables are updated before the optimizer step. Autograd
            # needs the multiplier snapshot used to build this objective.
            penalty+=(dual[name].clone()*value+.5*rho*value.square()).mean()
        # Keep the restored solution near the actual observed proposals, with
        # feasibility constraints taking precedence. Label arrays are absent.
        data=F.smooth_l1_loss((xyz-cache['baseline'])/.03,torch.zeros_like(xyz),beta=.5)
        if a.closest_proposal:
            targetpose=cache['baseline']-cache['baseline'][:,5:6]
            data+=.75*F.smooth_l1_loss(((xyz-xyz[:,5:6])-targetpose)/.03,torch.zeros_like(xyz),beta=.5)
        parameter_prior=.0001*(decoder.angles(local)-decoder.angles(seed['local'])).square().mean()
        loss=50*penalty+a.data_weight*data+parameter_prior
        assert torch.isfinite(loss)
        if step%10==0:
            with torch.no_grad():
                for name,value in constraints.items():dual[name].add_(rho*value.detach()).clamp_max_(10000)
                check_and_keep(world,dict(local=local,global_six=rotation,root=root,beta=beta))
        if step%100==0 or step==a.steps:
            entry=dict(step=step,accepted_tracks=int(accepted.sum()),tracks=groups,loss=float(loss.detach()),max_point_violation_mm=float(point.max().detach()*1000),seconds=time.time()-start)
            history.append(entry);save(out/'history.json',history);print(json.dumps(entry),flush=True)
        if step==a.steps:break
        optimizer.zero_grad(set_to_none=True);loss.backward()
        for parameter in [local,rotation,root,shape]:
            grad=parameter.grad;norm=grad.flatten(1).norm(dim=-1,keepdim=True).clamp_min(1e-8)
            grad.mul_((20/norm).clamp_max(1).reshape(len(grad),*[1]*(grad.ndim-1)))
        optimizer.step()
    with torch.no_grad():
        # Verify every stored accepted track using exactly the saved parameters.
        stored=decoder(chosen['local'],chosen['global_six'],chosen['root'],chosen['beta'],group,initial['sign'])
        sx,sp,sb,sm=violations(stored,cache,initial,LIMITS)
        take=accepted[group]
        assert not (sp[take]>1e-6).any();assert not (sb[take]>1e-6).any()
        if take.any():assert torch.allclose(output[take],sx[take],atol=1e-7,rtol=0)
        for name,value in sm.items():
            if value is None:continue
            frame=initial['pair'][0] if 'speed' in name else initial['triple'][0]
            assert not (value.norm(dim=-1)[accepted[group[frame]]]>LIMITS[name]+1e-5).any()
        torch.save(dict(prediction=output.cpu(),accepted=accepted.cpu(),frame_accepted=take.cpu(),parameters={k:v.cpu() for k,v in chosen.items()},group=group.cpu()),out/'results.pt')
    # Labels enter only here, after feasible parameters and acceptances freeze.
    labelsall=torch.load(source/'evaluation_labels.pt',weights_only=False,mmap=True);labels={k:v[ids] for k,v in labelsall.items()};sides=identities(rows,labels)
    old=torch.load(rootpath/'side_data_v16/consensus/dense_data.pt',weights_only=False,mmap=True);original=cache['base'].clone()
    for i,r in enumerate(rows):
        if r['window_index'] is not None:original[i]=old['original_base_for_evaluation'][r['window_index']].to(device)
    baseline=report(cache['baseline'],cache,labels,rows,sides,original)
    final=report(output,cache,labels,rows,sides,original,take);ok,conditions=eligible(final,baseline)
    save(out/'selection.json',dict(complete=True,baseline=baseline,final=final,eligible=ok,conditions=conditions,
        accepted_tracks=int(accepted.sum()),tracks=groups,default_changed=False,seconds=time.time()-start,
        accepted_parameter_recheck_passed=True,scope='Falling back retains v16, which is not structurally/time guaranteed; allframe metrics include these fallbacks. Feasibility alone is not adoption.'))
    print(json.dumps(dict(complete=True,eligible=ok,accepted_tracks=int(accepted.sum()),metrics=final['metrics'])),flush=True)


if __name__=='__main__':main()
