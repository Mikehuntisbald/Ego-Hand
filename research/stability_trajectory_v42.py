"""Observation-only, offline stability-first hand trajectories.

All outputs are decoded from bounded UmeTrack-compatible parameters. There
is no free-XYZ fallback, point clipping, or per-frame diffusion draw switch.
Hard guarantees cover the saved timestamps within each continuous segment,
not unseen frames, changing tracking IDs, collision freedom, or accuracy.
"""
import collections
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.spatial.transform import Rotation, Slerp
from joint_mano_model_v29 import rotation_to_six, six_to_rotation

DEFAULT = dict(root_speed=1.2, pose_speed=.65, root_acc=12., pose_acc=10.,
               wrist_speed=5., wrist_acc=30., joint_speed=6., joint_acc=40.,
               max_bridge_s=.55, nominal_hz=30., smoothing_s=.095,
               steps=350, shape_cap=2., rgb_weight=.035)


def exp_rotation(v):
    x,y,z=v.unbind(-1);q=torch.zeros_like(x)
    K=torch.stack([q,-z,y,z,q,-x,-y,x,q],-1).reshape(*v.shape[:-1],3,3)
    a=v.norm(dim=-1)
    A=torch.sinc(a/math.pi)[...,None,None]
    B=(.5*torch.sinc(a/(2*math.pi)).square())[...,None,None]
    return torch.eye(3,device=v.device,dtype=v.dtype)+A*K+B*(K@K)


def dense_layout(rows, max_bridge_s=.55, nominal_hz=30.):
    """Densify existing IDs only; never associate tracks using labels."""
    tracks=collections.defaultdict(list)
    for i,r in enumerate(rows):
        tracks[(r['sequence'],r['clip'],r['track_id'])].append(i)
    segments=[];source_map=np.empty(len(rows),dtype=np.int64)
    left=[];right=[];fraction=[];times=[];track_group=[];keys=sorted(tracks)
    for g,key in enumerate(keys):
        indices=sorted(tracks[key],key=lambda i:rows[i]['timestamp_ns'])
        current=[]
        for j,i in enumerate(indices):
            ti=rows[i]['timestamp_ns']/1e9
            if j:
                previous=indices[j-1];tp=rows[previous]['timestamp_ns']/1e9;dt=ti-tp
                if dt<=0:raise ValueError('Track timestamps must be strictly increasing')
                if dt>max_bridge_s:
                    segments.append(current);current=[]
                else:
                    intervals=max(1,int(round(dt*nominal_hz)))
                    for k in range(1,intervals):
                        w=k/intervals;current.append(len(left))
                        left.append(previous);right.append(i);fraction.append(w)
                        times.append(tp+w*dt);track_group.append(g)
            source_map[i]=len(left);current.append(len(left))
            left.append(i);right.append(i);fraction.append(0.)
            times.append(ti);track_group.append(g)
        if current:segments.append(current)
    return dict(left=np.asarray(left),right=np.asarray(right),fraction=np.asarray(fraction),
                times=np.asarray(times),group=np.asarray(track_group),keys=keys,
                source_map=source_map,segments=segments)


def derivatives(layout,device):
    pairs=[];triples=[]
    for segment in layout['segments']:
        for a,b in zip(segment,segment[1:]):
            pairs.append((a,b,layout['times'][b]-layout['times'][a]))
        for a,b,c in zip(segment,segment[1:],segment[2:]):
            triples.append((a,b,c,layout['times'][b]-layout['times'][a],layout['times'][c]-layout['times'][b]))
    def convert(values,count):
        return tuple(torch.tensor([x[k] for x in values],device=device,
                                  dtype=torch.long if k<count else torch.float32)
                     for k in range(count+(1 if count==2 else 2))) if values else None
    return convert(pairs,2),convert(triples,3)


def difference(x,pair,triple):
    v=a=None
    if pair is not None:
        i,j,dt=pair;v=(x[j]-x[i])/dt.reshape(-1,*([1]*(x.ndim-1)))
    if triple is not None:
        i,j,k,d1,d2=triple;shape=(-1,*([1]*(x.ndim-1)))
        a=((x[k]-x[j])/d2.reshape(shape)-(x[j]-x[i])/d1.reshape(shape))/((d1+d2)/2).reshape(shape)
    return v,a


def robust_smooth(y,t,weights,tau):
    """IRLS nonuniform-time acceleration regularization, without labels."""
    n=len(y)
    if n<3:return y.copy()
    dt=np.diff(t);a=np.arange(n-2);d1=dt[:-1];d2=dt[1:];h=(d1+d2)/2
    D=sparse.coo_matrix((np.concatenate([1/d1/h,-(1/d1+1/d2)/h,1/d2/h]),
                        (np.tile(a,3),np.concatenate([a,a+1,a+2]))),shape=(n-2,n)).tocsr()
    base=np.clip(weights,.02,1.);w=base.copy();out=y.copy()
    for _ in range(3):
        out=spsolve(sparse.diags(w)+(tau**4)*(D.T@D)+sparse.eye(n)*1e-8,w[:,None]*y)
        if out.ndim==1:out=out[:,None]
        error=np.linalg.norm((out-y)/(np.std(y,axis=0).clip(.03)[None]),axis=-1)
        w=base*np.minimum(1.,2./np.maximum(error,1e-8))
    return out


def trajectory_motion(world,rotations,angles,pair,triple):
    pose=world-world[:,5:6];rs,ra=difference(world[:,5],pair,triple)
    ps,pa=difference(pose,pair,triple);js,ja=difference(angles,pair,triple)
    values=dict(root_speed=rs,root_acc=ra,pose_speed=ps,pose_acc=pa,
                joint_speed=js,joint_acc=ja,wrist_speed=None,wrist_acc=None)
    if pair is not None:
        a,b,dt=pair
        delta=rotations[b]@rotations[a].transpose(-1,-2)
        # atan2 is stable around identity; true world-frame rotation increments.
        skew=torch.stack([delta[:,2,1]-delta[:,1,2],delta[:,0,2]-delta[:,2,0],delta[:,1,0]-delta[:,0,1]],-1)*.5
        sine=skew.norm(dim=-1);cosine=((delta.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
        theta=torch.atan2(sine,cosine)
        velocity=skew*(theta/sine.clamp_min(1e-7))[:,None]/dt[:,None]
        # Exact pi is unusual but cannot be allowed to masquerade as zero.
        near_pi=(cosine<-.9999)&(sine<1e-6)
        if near_pi.any():velocity=velocity.clone();velocity[near_pi,0]=math.pi/dt[near_pi]
        values['wrist_speed']=velocity
        if triple is not None:
            # Consecutive pair indices are laid out segment by segment.
            lookup={(int(i),int(j)):k for k,(i,j) in enumerate(zip(a.cpu(),b.cpu()))}
            ia,ib,ic,d1,d2=triple
            l=torch.tensor([lookup[(int(i),int(j))] for i,j in zip(ia.cpu(),ib.cpu())],device=world.device)
            r=torch.tensor([lookup[(int(j),int(k))] for j,k in zip(ib.cpu(),ic.cpu())],device=world.device)
            values['wrist_acc']=(velocity[r]-velocity[l])/((d1+d2)/2)[:,None]
    return values


def magnitude(name,value):
    return value.abs() if name.startswith('joint_') else value.norm(dim=-1)


def saved_check(decoder,parameters,layout,config):
    device=parameters['root'].device;pair,triple=derivatives(layout,device)
    world=decoder(parameters['local'],parameters['global_six'],parameters['root'],
                  parameters['beta'],parameters['group'],parameters['sign'])
    rotations=six_to_rotation(parameters['global_six']);angles=decoder.angles(parameters['local'])[:,:20]
    moves=trajectory_motion(world,rotations,angles,pair,triple)
    maxima={k:float(magnitude(k,v).max()) if v is not None and v.numel() else 0. for k,v in moves.items()}
    lo,hi=decoder.joint_limits[:20].unbind(-1)
    rotation_error=float((rotations.transpose(-1,-2)@rotations-torch.eye(3,device=device)).abs().max())
    passed=all(maxima[k]<=config[k]+2e-4 for k in maxima)
    passed &= bool(torch.isfinite(world).all() and (angles>=lo-1e-6).all() and (angles<=hi+1e-6).all())
    passed &= rotation_error<2e-5 and bool((torch.linalg.det(rotations)>.9999).all())
    return world,dict(passed=bool(passed),maxima=maxima,limits={k:config[k] for k in maxima},
                      dense_frames=len(world),original_frames=len(layout['source_map']),
                      interpolated_frames=len(world)-len(layout['source_map']),segments=len(layout['segments']),
                      finite=bool(torch.isfinite(world).all()),rotation_error=rotation_error,raw_xyz_fallback_frames=0,
                      guarantee_scope='Saved timestamps within the same predicted-ID continuous segment; no collision or correctness guarantee')


def fit_stable(decoder,cache,obs,rows,config=None,progress=None):
    cfg=dict(DEFAULT,**(config or {}));device=cache['base'].device
    if 'heat_xy' in cache and cache['heat_xy'].shape!=(len(rows),20,2):
        raise ValueError('Current-frame RGB localization must have shape Nx20x2')
    initial=decoder.initialize(obs,cache,rows)
    layout=dense_layout(rows,cfg['max_bridge_s'],cfg['nominal_hz'])
    left,right,w=layout['left'],layout['right'],layout['fraction']
    group=torch.tensor(layout['group'],device=device);source=torch.tensor(layout['source_map'],device=device)
    sign=initial['sign'][torch.tensor(left,device=device)]
    beta=initial['beta'].clamp(-cfg['shape_cap'],cfg['shape_cap']).detach()
    R=six_to_rotation(initial['global_six']).detach().cpu().numpy()
    theta=decoder.angles(initial['local'])[:,:20].detach().cpu().numpy()
    roots=initial['root'].detach().cpu().numpy()
    risk=cache.get('risk',torch.zeros(len(rows),20,2,device=device)).detach().cpu().numpy()
    confidence=np.clip(1-risk[:,:,1].mean(-1),.03,1.)
    root_conf=np.clip(1-risk[:,:,0].mean(-1),.03,1.)
    side_mismatch=(obs['right'].long()!=(initial['sign']<0).long()).detach().cpu().numpy()
    # A chirality vote cannot be applied to rotations/angles fitted for the
    # other hand unchanged. Treat such parameter observations as missing and
    # interpolate the consistent-side estimates in world/parameter space.
    original_group=initial['group'].detach().cpu().numpy()
    for g in range(len(initial['keys'])):
        ix=np.flatnonzero(original_group==g)
        ix=ix[np.argsort([rows[i]['timestamp_ns'] for i in ix])]
        good=ix[~side_mismatch[ix]];bad=ix[side_mismatch[ix]]
        if not len(bad):continue
        times=np.asarray([rows[i]['timestamp_ns']/1e9 for i in good])
        bt=np.asarray([rows[i]['timestamp_ns']/1e9 for i in bad])
        for k in range(20):theta[bad,k]=np.interp(bt,times,theta[good,k])
        for k in range(3):roots[bad,k]=np.interp(bt,times,roots[good,k])
        R[bad]=Slerp(times,Rotation.from_matrix(R[good]))(np.clip(bt,times[0],times[-1])).as_matrix() if len(good)>1 else R[good[0]]
    confidence[side_mismatch]*=.05;root_conf[side_mismatch]*=.05
    dense_theta=(1-w[:,None])*theta[left]+w[:,None]*theta[right]
    dense_root=(1-w[:,None])*roots[left]+w[:,None]*roots[right]
    rotation0=np.empty((len(left),3,3));vectors=np.empty((len(left),3));anchors=np.empty_like(rotation0)
    smooth_theta=dense_theta.copy();smooth_root=dense_root.copy()
    for segment in layout['segments']:
        ix=np.asarray(segment);actual=ix[left[ix]==right[ix]];oi=left[actual]
        # Select a medoid around the proper SO(3) mean, avoiding a high-score
        # rotational outlier as the logarithm chart origin.
        mean=Rotation.from_matrix(R[oi]).mean(weights=confidence[oi])
        distances=(mean.inv()*Rotation.from_matrix(R[oi])).magnitude()
        chosen=int(oi[np.argmin(distances)]);anchor=R[chosen]
        local=Rotation.from_matrix(np.einsum('ij,njk->nik',anchor.T,R[oi])).as_rotvec()
        t=layout['times'][actual];td=layout['times'][ix]
        rv=np.stack([np.interp(td,t,local[:,k]) for k in range(3)],-1)
        cw=np.interp(td,t,confidence[oi]);rw=np.interp(td,t,root_conf[oi])
        virtual=left[ix]!=right[ix];cw[virtual]*=.15;rw[virtual]*=.15
        smooth_theta[ix]=robust_smooth(dense_theta[ix],td,cw,cfg['smoothing_s'])
        smooth_root[ix]=robust_smooth(dense_root[ix],td,rw,cfg['smoothing_s'])
        vectors[ix]=robust_smooth(rv,td,cw,cfg['smoothing_s']);anchors[ix]=anchor
        rotation0[ix]=anchor[None]@Rotation.from_rotvec(rv).as_matrix()
    lo,hi=decoder.joint_limits[:20].unbind(-1)
    theta0=torch.tensor(smooth_theta,device=device,dtype=torch.float32)
    raw0=torch.logit(((theta0-lo)/(hi-lo)).clamp(.001,.999))
    local=nn.Parameter(raw0.clone());rv=nn.Parameter(torch.tensor(vectors,device=device,dtype=torch.float32))
    root=nn.Parameter(torch.tensor(smooth_root,device=device,dtype=torch.float32))
    anchor=torch.tensor(anchors,device=device,dtype=torch.float32)
    pair,triple=derivatives(layout,device)
    target=obs['xyz'];target_pose=target-target[:,5:6]
    measured=cache['base'];measured_pose=measured-measured[:,5:6]
    pweight=(1-cache.get('risk',torch.zeros(len(rows),20,2,device=device))[:,:,1]).clamp_min(.03)
    cweight=(1-cache.get('risk',torch.zeros(len(rows),20,2,device=device))[:,:,0]).clamp_min(.03)
    reliable_side=torch.tensor(np.where(side_mismatch,.05,1.),device=device,dtype=torch.float32)
    pweight=pweight*reliable_side[:,None];cweight=cweight*reliable_side[:,None]
    optimizer=torch.optim.Adam([dict(params=[local,rv],lr=.006),dict(params=[root],lr=.0005)])
    history=[]
    for step in range(cfg['steps']):
        rot=anchor@exp_rotation(rv);world=decoder(local,rotation_to_six(rot),root,beta,group,sign)
        current=world[source]
        camera=torch.einsum('njc,nck->njk',current-cache['translation'][:,None],cache['rotation'])
        pose=camera-camera[:,5:6]
        def hub(x):return F.smooth_l1_loss(x,torch.zeros_like(x),reduction='none',beta=.5).sum(-1)
        data=(cweight*hub((camera-target)/.03)+pweight*hub((pose-target_pose)/.02)).mean()
        data+=.3*(cweight*hub((camera-measured)/.03)+pweight*hub((pose-measured_pose)/.02)).mean()
        angular=decoder.angles(local)[:,:20]
        # Parameter-space prior retains observed articulation, rather than
        # forcing the entire clip into the neutral hand.
        prior=.02*((angular-theta0)/.5).square().mean()+.01*(rv-torch.tensor(vectors,device=device,dtype=torch.float32)).square().mean()
        temporal=world.sum()*0;penalty=world.sum()*0
        for name,x in [('root',root),('pose',world-world[:,5:6]),('joint',angular),('wrist',rv)]:
            velocity,acc=difference(x,pair,triple)
            if acc is not None:temporal+=(acc/(cfg[name+'_acc']*.4)).square().mean()
            for suffix,value in [('speed',velocity),('acc',acc)]:
                if value is not None:
                    mag=magnitude(name+'_'+suffix,value)
                    penalty+=(mag/(cfg[name+'_'+suffix]*.9)-1).clamp_min(0).square().mean()
        rgb=world.sum()*0
        if all(k in cache for k in ['heat_xy','camera_params','roi']):
            from hand3d_rollout_v8 import project_fisheye624
            uv=project_fisheye624(camera,cache['camera_params'])/1408
            size=(cache['roi'][:,2:]-cache['roi'][:,:2]).mean(-1).clamp_min(.01)
            rgb=(pweight*hub((uv-cache['heat_xy'])/size[:,None,None]*10)).mean()
        loss=data+prior+.08*temporal+100*penalty+cfg['rgb_weight']*rgb
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite stability objective')
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_([local,rv,root],10);optimizer.step()
        if step%100==0 or step==cfg['steps']-1:
            event=dict(step=step+1,loss=float(loss.detach()),data=float(data.detach()),temporal=float(temporal.detach()),penalty=float(penalty.detach()))
            history.append(event)
            if progress:progress(event)
    with torch.no_grad():
        rot=anchor@exp_rotation(rv);angles=decoder.angles(local)[:,:20]
        restoration=[]
        # Feasible deterministic fallback also uses parameters and shared shape.
        # Root has linear derivative constraints, so scalar contraction certifies
        # it exactly. Pose contraction is decoded and checked on every attempt.
        for segment in layout['segments']:
            ix=torch.tensor(segment,device=device);sub=dict(layout,segments=[list(range(len(ix)))],times=layout['times'][segment])
            pp,tt=derivatives(sub,device)
            rs,ra=difference(root[ix],pp,tt);ratio=1.
            for key,value in [('root_speed',rs),('root_acc',ra)]:
                if value is not None:ratio=max(ratio,float(magnitude(key,value).max())/(cfg[key]*.97))
            root_alpha=1/ratio;center=root[ix].mean(0);root[ix]=center+root_alpha*(root[ix]-center)
            th=angles[ix].clone();rr=rot[ix].clone();weight=torch.tensor(confidence[left[np.asarray(segment)]],device=device)
            th_anchor=(th*weight[:,None]).sum(0)/weight.sum()
            best=int(weight.argmax());r_anchor=rr[best]
            relative=Rotation.from_matrix((r_anchor.T@rr).cpu().numpy()).as_rotvec()
            relative=torch.tensor(relative,device=device,dtype=torch.float32)
            chosen=None
            for alpha in [1.,.95,.9,.8,.7,.6,.5,.4,.3,.2,.1,.05,.02,.01,0.]:
                ta=th_anchor+alpha*(th-th_anchor);r=r_anchor[None]@exp_rotation(relative*alpha)
                raw=torch.logit(((ta-lo)/(hi-lo)).clamp(1e-5,1-1e-5))
                world=decoder(raw,rotation_to_six(r),root[ix],beta,group[ix],sign[ix])
                moves=trajectory_motion(world,r,decoder.angles(raw)[:,:20],pp,tt)
                if all(v is None or float(magnitude(k,v).max())<=cfg[k]*.985 for k,v in moves.items()):
                    local[ix]=raw;rot[ix]=r;chosen=alpha;break
            if chosen is None:raise RuntimeError('Constant parameter fallback was unexpectedly infeasible')
            restoration.append(dict(frames=len(ix),root_scale=root_alpha,pose_scale=chosen,
                                    constant_pose_fallback=chosen==0.))
        parameters=dict(local=local.detach(),global_six=rotation_to_six(rot),root=root.detach(),beta=beta,
                        group=group,sign=sign)
        world,check=saved_check(decoder,parameters,layout,cfg)
        if not check['passed']:raise RuntimeError(str(check))
        prediction=torch.einsum('njc,nck->njk',world[source]-cache['translation'][:,None],cache['rotation'])
        confirmed=cache.get('confirmed',torch.zeros(len(rows),20,device=device,dtype=torch.bool))
        conflicts=confirmed&((prediction-cache['base']).norm(dim=-1)>1e-6)
        return dict(prediction=prediction,world=world,parameters=parameters,layout=layout,check=check,
                    config=cfg,history=history,restoration=restoration,manual_anchor_conflicts=conflicts,
                    side_parameter_outlier_frames=int(side_mismatch.sum()),
                    scope='Stability-first estimates, not accuracy-certified labels. Manual conflicts are review-only; never overwrite anchors.')
