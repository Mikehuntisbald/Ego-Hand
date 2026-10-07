"""Joint trajectory MAP solver; shared shape, hard acceptance, no XYZ clipping."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_rollout_v8 import project_fisheye624
from joint_mano_model_v29 import violations,point_radii
def fit_trajectory(decoder,cache,obs,rows,config,progress=None):
    """Observation-only MAP solve followed by track-level feasible selection."""
    initial=decoder.initialize(obs,cache,rows);group=initial['group'];n=len(rows);device=cache['base'].device
    beta0=initial['beta'].clamp(-3.999,3.999)
    initial['beta']=beta0
    local=nn.Parameter(initial['local'].clone());global_six=nn.Parameter(initial['global_six'].clone());root=nn.Parameter(initial['root'].clone())
    shape=nn.Parameter(torch.atanh(beta0/4))
    parameters=[dict(params=[local,global_six],lr=.012),dict(params=[root],lr=.0015),dict(params=[shape],lr=.01)]
    optimizer=torch.optim.Adam(parameters);limits=config['limits'];history=[]
    target=cache['baseline']+(cache['proposal']-cache['baseline'])*config.get('raw_mix',0.)
    target_pose=target-target[:,5:6];base=cache['base'];risk=cache['risk'];size=(cache['roi'][:,2:]-cache['roi'][:,:2]).mean(-1).clamp_min(.01)
    def huber(x):return F.smooth_l1_loss(x,torch.zeros_like(x),reduction='none',beta=.5)
    for step in range(config.get('steps',300)):
        beta=4*shape.tanh();world=decoder(local,global_six,root,beta,group,initial['sign'])
        xyz,point,bend,moves=violations(world,cache,initial,limits)
        data=(huber((xyz-target)/.03).sum(-1)+.75*huber(((xyz-xyz[:,5:6])-target_pose)/.03).sum(-1)).mean()
        anch=((.05+.95*(1-risk[:,:,0]))*huber((xyz-base)/.03).sum(-1)).mean()
        uv=project_fisheye624(xyz,cache['camera_params'])/1408
        rgb=huber((uv-cache['heat_xy'])/size[:,None,None]*10).sum(-1).mean()
        pose_prior=decoder.pose_prior(local,initial['local'])
        shape_prior=(beta-beta0).square().mean()
        overlap=world.sum()*0
        if 'fusion_world' in cache:
            fused=cache['fusion_world'];fpose=fused-fused[:,5:6]
            overlap=huber((world-fused)/.03).sum(-1).mean()+.75*huber(((world-world[:,5:6])-fpose)/.03).sum(-1).mean()
        temporal=world.sum()*0;hard=world.sum()*0
        if moves['root_acc'] is not None:
            temporal=huber(moves['root_acc']/10).sum(-1).mean()+.5*huber(moves['pose_acc']/10).sum(-1).mean()
        # Soft motion terms guide fitting. Hard limits are checked jointly at
        # acceptance. Infeasible low-score tracks must not dominate all other
        # tracks' gradients through an unbounded penalty and global clipping.
        if config.get('hard_motion_loss',False):
            for name in ['root_speed','pose_speed','root_acc','pose_acc']:
                v=moves[name]
                if v is not None:hard+=((v.norm(dim=-1)/limits[name]-1).clamp_min(0)).square().mean()
        protect=(point/.005).square().mean()
        loss=data+.1*anch+config.get('rgb_weight',.05)*rgb+.002*pose_prior+.005*shape_prior+config.get('temporal_weight',.02)*temporal+config.get('overlap_weight',.1)*overlap+20*protect+10*bend.square().mean()+20*hard
        assert torch.isfinite(loss),'Nonfinite MAP objective'
        optimizer.zero_grad(set_to_none=True);loss.backward()
        for parameter in [local,global_six,root,shape]:
            gradient=parameter.grad
            scale=(10/gradient.reshape(len(gradient),-1).norm(dim=-1,keepdim=True).clamp_min(1e-8)).clamp_max(1)
            gradient.mul_(scale.reshape(len(gradient),*[1]*(gradient.ndim-1)))
        optimizer.step()
        if step%100==0 or step==config.get('steps',300)-1:
            history.append(dict(step=step+1,loss=float(loss.detach()),data=float(data.detach()),rgb=float(rgb.detach()),temporal=float(temporal.detach()),max_point_violation_mm=float(point.detach().max()*1000)))
            if progress is not None:progress(history[-1])
    with torch.no_grad():
        final=dict(local=local.detach(),global_six=global_six.detach(),root=root.detach(),beta=(4*shape.tanh()).detach())
        hypothesis=decoder(final['local'],final['global_six'],final['root'],final['beta'],group,initial['sign'])
        accepted=torch.zeros(len(initial['keys']),device=device,dtype=torch.bool);alpha=torch.full((len(accepted),),-1.,device=device)
        output=cache['baseline'].clone();parameter_output={k:v.clone() for k,v in final.items()};diagnostic=[]
        # A single interpolation strength per whole track; never clip XYZ.
        for strength in [1.,.75,.5,.25,.125,.0625,0.]:
            trial={k:initial[k]+strength*(final[k]-initial[k]) for k in ['local','global_six','root','beta']}
            world=decoder(trial['local'],trial['global_six'],trial['root'],trial['beta'],group,initial['sign'])
            xyz,point,bend,moves=violations(world,cache,initial,limits)
            framebad=(point>1e-6).any(-1)|(bend>1e-6).any(-1)|~torch.isfinite(xyz).all(dim=(-1,-2))
            confirmed=cache.get('confirmed')
            if confirmed is not None and confirmed.any():
                # Exact locks and a parametric model can be incompatible.
                # Reject the whole track instead of copying a vertex afterward.
                framebad|=((xyz!=cache['base']).any(-1)&confirmed).any(-1)
            bad_count=torch.zeros(len(accepted),device=device,dtype=torch.long)
            bad_count.scatter_add_(0,group,framebad.long())
            if config.get('enforce_motion',True):
                for name in ['root_speed','pose_speed','root_acc','pose_acc']:
                    if moves[name] is None:continue
                    exceed=moves[name].norm(dim=-1)>limits[name]+1e-5
                    while exceed.ndim>1:exceed=exceed.any(-1)
                    index=initial['pair'][0] if 'speed' in name else initial['triple'][0]
                    bad_count.scatter_add_(0,group[index],exceed.long())
            groupbad=bad_count>0
            newly=~accepted&~groupbad
            take=newly[group];output[take]=xyz[take];accepted|=newly;alpha[newly]=strength
            for k in ['local','global_six','root']:parameter_output[k][take]=trial[k][take]
            parameter_output['beta'][newly]=trial['beta'][newly]
            diagnostic.append(dict(alpha=strength,new_tracks=int(newly.sum()),accepted_tracks=int(accepted.sum())))
        review_xyz=torch.einsum('njc,nck->njk',hypothesis-cache['translation'][:,None],cache['rotation'])
        rc,rr,confirmed=point_radii(cache)
        if confirmed.any():assert torch.equal(output[confirmed],cache['baseline'][confirmed]),'Confirmed XYZ may never be silently overwritten'
        return dict(prediction=output,hypothesis=review_xyz,parameters=parameter_output,accepted=accepted,frame_accepted=accepted[group],group=group,group_keys=initial['keys'],alpha=alpha,
            history=history,backtracking=diagnostic,limits=limits,scope='Accepted trajectories pass structure/point/motion guards. Rejectedtracks retain v16; hypothesis reviewonly. No GT selection, no freeXYZ clipping.')

