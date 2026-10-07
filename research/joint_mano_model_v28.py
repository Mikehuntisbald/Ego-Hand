"""Shared-shape differentiable MANO trajectories with joint-space acceptance.

No GT accepted. Pointwise XYZ clipping is deliberately absent. Infeasible
tracks retain their previous outputs and expose a separate review hypothesis.
"""
import collections,math
import torch
from torch import nn
from torch.nn import functional as F
from hand3d_rollout_v8 import project_fisheye624
import spatial_rgb_common as s

CHAINS=[[5,6,7,0],[5,8,9,10,1],[5,11,12,13,2],[5,14,15,16,3],[5,17,18,19,4]]
EDGES=[(u,v) for c in CHAINS for u,v in zip(c,c[1:])]

def six_to_rotation(x):
    a=F.normalize(x[...,:3],dim=-1,eps=1e-8)
    b=x[...,3:]-(a*x[...,3:]).sum(-1,keepdim=True)*a
    b=F.normalize(b,dim=-1,eps=1e-8);c=torch.linalg.cross(a,b,dim=-1)
    return torch.stack([a,b,c],-1)

def rotation_to_six(x):return x[...,:,:2].transpose(-1,-2).reshape(*x.shape[:-2],6)

@torch.no_grad()
def fuse_overlapping_trajectories(cache,inputs,rows):
    """Pool measured proposals, not final joints; MANO still generates output."""
    device=cache['base'].device
    lookup={}
    for i,r in enumerate(rows):lookup[r['primary_fid']]=i;lookup[r['center_fid']]=i
    src=[];slot=[];dest=[];weights=[]
    fids=inputs['feature_ids'].cpu();dt=inputs['dt'].cpu()
    for i,r in enumerate(rows):
        for t,fid in enumerate(fids[i].tolist()):
            j=lookup.get(fid)
            if j is None or abs(float(dt[i,t]))>.101:continue
            z=rows[j]
            if (z['sequence'],z['clip'],z['track_id'])!=(r['sequence'],r['clip'],r['track_id']):continue
            src.append(i);slot.append(t);dest.append(j);weights.append(math.exp(-abs(float(dt[i,t]))/.1))
    src=torch.tensor(src,device=device);slot=torch.tensor(slot,device=device);dest=torch.tensor(dest,device=device)
    weights=torch.tensor(weights,device=device,dtype=torch.float32)
    trajectory=cache['trajectory'].clone();trajectory[:,8]=cache['proposal']
    world=torch.einsum('njc,nkc->njk',trajectory[src,slot],cache['rotation'][src])+cache['translation'][src,None]
    sums=torch.zeros_like(cache['base']);count=torch.zeros(len(rows),device=device)
    sums.index_add_(0,dest,world*weights[:,None,None]);count.index_add_(0,dest,weights)
    assert (count>0).all()
    return sums/count[:,None,None]

def bending_cos(x):
    values=[]
    for chain in CHAINS:
        for u,v,w in zip(chain,chain[1:],chain[2:]):
            values.append(F.cosine_similarity(x[:,v]-x[:,u],x[:,w]-x[:,v],dim=-1,eps=1e-8))
    return torch.stack(values,-1)

def temporal_indices(rows,device):
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[(r['sequence'],r['clip'],r['track_id'])].append(i)
    ordered=sorted(groups);gid=torch.empty(len(rows),device=device,dtype=torch.long);pairs=[];triples=[]
    for g,key in enumerate(ordered):
        ix=sorted(groups[key],key=lambda i:rows[i]['timestamp_ns']);gid[ix]=g
        for a,b in zip(ix,ix[1:]):
            dt=(rows[b]['timestamp_ns']-rows[a]['timestamp_ns'])/1e9
            if 0<dt<=.101:pairs.append((a,b,dt))
        for a,b,c in zip(ix,ix[1:],ix[2:]):
            d1=(rows[b]['timestamp_ns']-rows[a]['timestamp_ns'])/1e9;d2=(rows[c]['timestamp_ns']-rows[b]['timestamp_ns'])/1e9
            if 0<d1<=.101 and 0<d2<=.101:triples.append((a,b,c,d1,d2))
    def tensor(xs,cols):
        return tuple(torch.tensor([x[c] for x in xs],device=device,dtype=torch.long if c<cols else torch.float32) for c in range(cols+(1 if cols==2 else 2)))
    pair=tensor(pairs,2) if pairs else None;triple=tensor(triples,3) if triples else None
    return gid,ordered,pair,triple

class ManoWorldDecoder(nn.Module):
    def __init__(self,device):
        super().__init__();model,_=s.common.load_model(device);self.mano=model.mano;del model
        for p in self.mano.parameters():p.requires_grad_(False)

    def forward(self,local_six,global_six,root,beta,group,sign):
        n=len(root);identity=torch.eye(3,device=root.device)[None,None].expand(n,1,3,3)
        out=self.mano(global_orient=identity,hand_pose=six_to_rotation(local_six),betas=beta[group],pose2rot=False)
        local=out.joints[:,s.common.MAPPING];local=local-local[:,5:6]
        local=local*torch.stack([sign,torch.ones_like(sign),torch.ones_like(sign)],-1)[:,None]
        world=torch.einsum('nij,nkj->nki',six_to_rotation(global_six),local)+root[:,None]
        return world

    def initialize(self,obs,cache,rows):
        device=cache['base'].device;group,keys,pair,triple=temporal_indices(rows,device)
        sign=obs['right'].float()*2-1;reflect=torch.diag_embed(torch.stack([sign,torch.ones_like(sign),torch.ones_like(sign)],-1))
        R=cache['rotation'];global_rot=R@obs['transform']@obs['global_orient'].reshape(-1,3,3)@reflect
        root=torch.einsum('njc,nkc->njk',obs['xyz'][:,5:6],R)[:,0]+cache['translation']
        local=rotation_to_six(obs['hand_pose'].reshape(-1,15,3,3));global_six=rotation_to_six(global_rot)
        beta=torch.stack([obs['betas'][group==g].median(0).values for g in range(len(keys))])
        return dict(local=local,global_six=global_six,root=root,beta=beta,group=group,keys=keys,pair=pair,triple=triple,sign=sign)

    @torch.no_grad()
    def reconstruction_check(self,obs,cache,rows):
        initial=self.initialize(obs,cache,rows);n=len(rows)
        out=self(initial['local'],initial['global_six'],initial['root'],obs['betas'],torch.arange(n,device=cache['base'].device),initial['sign'])
        expected=torch.einsum('njc,nkc->njk',obs['xyz'],cache['rotation'])+cache['translation'][:,None]
        err=(out-expected).norm(dim=-1)*1000
        return dict(mean_mm=float(err.mean()),max_mm=float(err.max()),passed=bool(err.max()<.02))

def movement(world,initial):
    pair=initial['pair'];triple=initial['triple'];zero=world.sum()*0;pose=world-world[:,5:6]
    values=dict(root_speed=None,pose_speed=None,root_acc=None,pose_acc=None)
    if pair is not None:
        a,b,dt=pair;values['root_speed']=(world[b,5]-world[a,5])/dt[:,None]
        values['pose_speed']=(pose[b]-pose[a])/dt[:,None,None]
    if triple is not None:
        a,b,c,d1,d2=triple;scale=(d1+d2)/2
        values['root_acc']=((world[c,5]-world[b,5])/d2[:,None]-(world[b,5]-world[a,5])/d1[:,None])/scale[:,None]
        values['pose_acc']=((pose[c]-pose[b])/d2[:,None,None]-(pose[b]-pose[a])/d1[:,None,None])/scale[:,None,None]
    return values

def point_radii(cache):
    policy=cache['policy'];risk=cache['risk'];large=policy['large_cap_m']
    rc=torch.where(risk[:,:,0]>=policy['camera_risk_threshold'],large,.0095)
    rr=torch.where(risk[:,:,1]>=policy['relative_risk_threshold'],large,.0095)
    confirmed=cache.get('confirmed',torch.zeros_like(rc,dtype=torch.bool))
    rc=torch.where(confirmed,0.,rc);rr=torch.where(confirmed,.0095,rr)
    return rc,rr,confirmed

def violations(world,cache,initial,limits):
    xyz=torch.einsum('njc,nck->njk',world-cache['translation'][:,None],cache['rotation'])
    base=cache['base'];rc,rr,confirmed=point_radii(cache)
    dc=(xyz-base).norm(dim=-1);dr=((xyz-xyz[:,5:6])-(base-base[:,5:6])).norm(dim=-1)
    point=torch.maximum((dc-rc).clamp_min(0),(dr-rr).clamp_min(0))
    cos=bending_cos(xyz);bend=(math.cos(math.radians(limits['max_bend_degrees']))-cos).clamp_min(0)
    moves=movement(world,initial);return xyz,point,bend,moves

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
        pose_prior=(six_to_rotation(local)-six_to_rotation(initial['local'])).square().mean()
        shape_prior=(beta-beta0).square().mean()
        overlap=world.sum()*0
        if 'fusion_world' in cache:
            fused=cache['fusion_world'];fpose=fused-fused[:,5:6]
            overlap=huber((world-fused)/.03).sum(-1).mean()+.75*huber(((world-world[:,5:6])-fpose)/.03).sum(-1).mean()
        temporal=world.sum()*0;hard=world.sum()*0
        if moves['root_acc'] is not None:
            temporal=huber(moves['root_acc']/10).sum(-1).mean()+.5*huber(moves['pose_acc']/10).sum(-1).mean()
        for name in ['root_speed','pose_speed','root_acc','pose_acc']:
            v=moves[name]
            if v is not None:hard+=((v.norm(dim=-1)/limits[name]-1).clamp_min(0)).square().mean()
        protect=(point/.005).square().mean()
        loss=data+.1*anch+config.get('rgb_weight',.05)*rgb+.002*pose_prior+.005*shape_prior+config.get('temporal_weight',.02)*temporal+config.get('overlap_weight',.1)*overlap+20*protect+10*bend.square().mean()+20*hard
        assert torch.isfinite(loss),'Nonfinite MAP objective'
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_([local,global_six,root,shape],10.);optimizer.step()
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
