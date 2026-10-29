"""Native MANO fixed-camera BA, with an explicit frozen-loss control profile."""
import numpy as np
from .errors import RuntimeFailure
from .clock import motion_pairs
from .geometry import project, matrix_rotvec, rotvec_matrix


def optimize(seeds, observations, codec, config, *, disable_additions=False, guard=None):
    from .precision import geometry_precision
    with geometry_precision(codec.device):
        return _optimize(seeds,observations,codec,config,disable_additions=disable_additions,guard=guard)


def _optimize(seeds, observations, codec, config, *, disable_additions=False, guard=None):
    import torch
    from torch.nn import functional as F
    from .native import project_parameters, exp_so3
    device = codec.device
    tensor = lambda x: torch.as_tensor(np.asarray(x), device=device, dtype=torch.float32)
    state0 = tensor([r['state'] for r in seeds])
    tracks = [r['track_id'] for r in seeds]
    unique = list(dict.fromkeys(tracks))
    groups = np.array([unique.index(t) for t in tracks])
    beta0 = tensor([np.median([r['betas'] for r in seeds if r['track_id']==t], axis=0) for t in unique])
    group_t = torch.tensor(groups, device=device)
    right = torch.tensor([r['right'] for r in seeds], device=device)
    stamps = np.array([r['timestamp_ns'] for r in seeds], np.int64)
    pairs, triples = motion_pairs(stamps, tracks, [r['segment'] for r in seeds], [True]*len(seeds), config['max_gap_ns'])
    pair = torch.tensor(pairs, device=device)
    triple = torch.tensor(triples, device=device)
    dp = tensor([(int(stamps[b])-int(stamps[a]))/1e9 for a, b in pairs])[:, None, None]
    dt0 = tensor([(int(stamps[b])-int(stamps[a]))/1e9 for a, b, c in triples])[:, None, None]
    dt1 = tensor([(int(stamps[c])-int(stamps[b]))/1e9 for a, b, c in triples])[:, None, None]
    # Physical anchors initialize along the seed-camera ray. Their surface-to-wrist
    # semantics were established upstream; surface depths never enter as wrist Z.
    state = torch.nn.Parameter(state0.clone())
    shape = torch.nn.Parameter(beta0.clone())
    optimizer = torch.optim.LBFGS([state, shape], lr=.7, max_iter=config['iterations'],
                                   line_search_fn='strong_wolfe', history_size=15)
    prepared = []
    for model in dict.fromkeys(o['camera'].model for o in observations):
        selected=[o for o in observations if o['camera'].model==model]
        T=tensor([o['world_from_camera'] for o in selected])
        prepared.append(dict(model=model,indices=torch.tensor([o['state_index'] for o in selected],device=device),
            R=T[:,:3,:3],T=T[:,:3,3],uv=tensor([o['uv'] for o in selected]),
            mask=tensor([o['point_valid'] for o in selected]),params=tensor([o['camera'].parameters for o in selected]),
            scale=tensor([max(float(o.get('box_scale_px',1)),1) for o in selected])[:,None,None],
            anchor=tensor([float(o.get('depth_m') or 0.) for o in selected]),
            depth_weight=tensor([float(o.get('depth_weight',1.)) if o.get('depth_valid') and o.get('depth_m') is not None else 0. for o in selected]),
            confidence=tensor([float(o.get('confidence',0)) for o in selected])))
    history = []

    def evaluate():
        with torch.autocast(device_type=device.type, enabled=False):
            world = codec.tensor(state, shape[group_t], right)
            reprojection, depth, depth_weight, preservation = [], [], [], []
            for group in prepared:
                cam = torch.einsum('nkj,nji->nki',world[group['indices']]-group['T'][:,None],group['R'])
                projected = project_parameters(cam,group['params'],group['model'])
                mask=group['mask']
                if disable_additions or not config['mask_outside_image']:
                    mask = torch.ones_like(mask)
                per=F.smooth_l1_loss((projected-group['uv'])/group['scale'],torch.zeros_like(group['uv']),reduction='none')
                reprojection.append((per*mask[...,None]).sum((1,2))/(mask.sum(1)*2).clamp_min(1.))
                depth.append(F.smooth_l1_loss((cam[:,0,2]-group['anchor'])/.02,torch.zeros_like(group['anchor']),reduction='none')*group['depth_weight'])
                depth_weight.append(group['depth_weight'])
                # This is prediction confidence, never a GT accurate-point mask.
                preservation.append(per.mean((1,2))*group['confidence'])
            zero = state.sum()*0
            reproj = torch.cat(reprojection).mean()
            depth_loss = torch.cat(depth).sum()/torch.cat(depth_weight).sum().clamp_min(1.)
            prior = (state[:, 6:]-state0[:, 6:]).square().mean()
            root_prior = (state[:, 3:6]-state0[:, 3:6]).square().mean()
            position_prior = ((state[:, :3]-state0[:, :3])/.1).square().mean()
            speed, accel, angular = zero, zero, zero
            if len(pairs):
                v = (world[pair[:, 1]]-world[pair[:, 0]])/(60*dp)
                speed = F.smooth_l1_loss(v/.01, torch.zeros_like(v))
                rotations = exp_so3(state[:, 3:6])
                angular = ((rotations[pair[:, 1]]-rotations[pair[:, 0]])/(60*dp)).square().mean()
            if len(triples):
                v0 = (world[triple[:, 1]]-world[triple[:, 0]])/(60*dt0)
                v1 = (world[triple[:, 2]]-world[triple[:, 1]])/(60*dt1)
                a = (v1-v0)/(60*(dt0+dt1)/2)
                accel = F.smooth_l1_loss(a/.005, torch.zeros_like(a))
            # These legacy weights are retained ONLY with their original units
            # and normalizations to make the frozen-loss control reproducible.
            total = (20*reproj+5*depth_loss+.08*prior+.05*root_prior+.01*position_prior+
                     .002*speed+.02*accel+.002*angular+.02*(shape-beta0).square().mean())
            if not disable_additions:
                if len(pairs):
                    rotations = exp_so3(state[:,3:6])
                    relative = rotations[pair[:,0]].transpose(-1,-2)@rotations[pair[:,1]]
                    skew = torch.stack([relative[:,2,1]-relative[:,1,2],relative[:,0,2]-relative[:,2,0],
                                        relative[:,1,0]-relative[:,0,1]],-1)/2
                    sine = skew.square().sum(-1).clamp_min(1e-16).sqrt()
                    cosine = (relative.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
                    angle = torch.atan2(sine,cosine)
                    angular_velocity = angle/(60*dp[:,0,0])
                    total += config['world_root_so3_weight']*F.smooth_l1_loss(angular_velocity/.05,torch.zeros_like(angle))
                    wrist_velocity = (world[pair[:, 1], :1]-world[pair[:, 0], :1])/(60*dp)
                    total += config['wrist_motion_weight']*F.smooth_l1_loss(wrist_velocity/.01, torch.zeros_like(wrist_velocity))
                total += config['observation_preservation_weight']*torch.cat(preservation).mean()
            return total, world, dict(reprojection=float(reproj.detach()), depth=float(depth_loss.detach()),
                                     speed=float(speed.detach()), acceleration=float(accel.detach()))

    def closure():
        if guard is not None:guard()
        optimizer.zero_grad()
        loss, _, parts = evaluate()
        if not torch.isfinite(loss):
            raise RuntimeFailure('BA_nonfinite', 'Native BA objective is non-finite')
        loss.backward()
        if state.grad is None or not torch.isfinite(state.grad).all() or not torch.isfinite(shape.grad).all():
            raise RuntimeFailure('BA_gradient', 'Native BA gradient is invalid')
        history.append(dict(objective=float(loss.detach()), **parts))
        return loss

    before = float(closure().detach())
    optimizer.step(closure)
    with torch.no_grad():
        final, world, _ = evaluate()
        for group in prepared:
            camera=torch.einsum('nkj,nji->nki',world[group['indices']]-group['T'][:,None],group['R'])
            if not torch.isfinite(camera).all() or (camera[:,:,2]<=.01).any():
                raise RuntimeFailure('BA_geometry','Native BA produced nonfinite or behind-camera joints; result is not delivered')
    return dict(state=state.detach().cpu().numpy(), betas=shape[group_t].detach().cpu().numpy(),
                joints=world.cpu().numpy(), right=np.array([r['right'] for r in seeds]),
                before=before, after=float(final), history=history,
                loss_profile='frozen_native_v1', fixed_camera=True, new_losses_disabled=disable_additions)
