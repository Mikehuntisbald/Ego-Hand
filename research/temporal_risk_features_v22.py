"""Per-frame per-finger reliability features; observation inputs only."""
import torch

def features(data, ids):
    f = data['feature_ids'][ids]
    time = data['dt'][ids]
    available = data['available_bank'][f] & (f>0)[...,None]
    xy = torch.where(data['observed_2d'][ids][...,None],data['xy'][ids],0.)
    positions = data['positions_bank'][f]
    rgb = data['risk_rgb_bank'][f].float()
    own = data['xyz_camera_bank'][f]
    own_pose = own-own[:,:,5:6]
    world = data['world'][f]
    B,T,J,_ = own.shape
    eye = torch.eye(J,device=f.device)[None].expand(B,-1,-1)
    rows = torch.arange(B,device=f.device)[:,None]
    joints = torch.arange(J,device=f.device)[None]
    out = []
    for center in range(T):
        R = data['rotation'][f[:,center]]
        translation = data['translation'][f[:,center]]
        aligned = torch.einsum('btjc,bck->btjk',world-translation[:,None,None],R)
        dt = time-time[:,center: center+1]
        base = own[:,center]
        parts = [base/.5,own_pose[:,center]/.1,available[:,center,:,None].float(),
            data['observed_2d'][ids,center,:,None].float(),
            data['scores'][f[:,center]][:,None,None].expand(-1,J,1),eye]
        for radius in [0.,.1,.5]:
            valid = available & (dt.abs()>radius)[:,:,None]
            valid[:,center] = False
            delta = dt[:,:,None].expand(-1,-1,J)
            left = delta.masked_fill(~(valid & (delta<0)),float('-inf')).argmax(1)
            right = delta.masked_fill(~(valid & (delta>0)),float('inf')).argmin(1)
            has_left = (valid & (delta<0)).any(1)
            has_right = (valid & (delta>0)).any(1)
            a,z = aligned[rows,left,joints],aligned[rows,right,joints]
            ta,tz = dt.gather(1,left),dt.gather(1,right)
            weight = (-ta/(tz-ta).clamp_min(1e-5)).clamp(0,1)
            both = has_left & has_right
            either = has_left | has_right
            estimate = torch.where(both[...,None],a*(1-weight[...,None])+z*weight[...,None],
                torch.where(has_left[...,None],a,z))
            parts.extend([((base-estimate)/.1).clamp(-10,10)*either[...,None],both[...,None].float(),either[...,None].float()])
        # RGB describes each neighboring frame's own projected predicted point.
        # A projected point is a query location, never finger visibility truth.
        for offset in [-.1,0.,.1]:
            distance = (dt-offset).abs().masked_fill(f<=0,float('inf'))
            neighbor = distance.argmin(1)
            row = torch.arange(B,device=f.device)
            loc = positions[row,neighbor]
            query = xy[row,neighbor]
            selected_rgb = rgb[row,neighbor]
            size = (data['roi'][f[row,neighbor],2:]-data['roi'][f[row,neighbor],:2]).mean(-1).clamp_min(.01)
            dist = ((loc[:,None]-query[:,:,None])/size[:,None,None,None]).square().sum(-1)
            weight = (-dist/(2*(1/12)**2)).softmax(-1)
            exists = (f[row,neighbor]>0) & (distance[row,neighbor]<=1.)
            parts.append(torch.einsum('bjs,bsc->bjc',weight,selected_rgb)*exists[:,None,None])
        parts.append(rgb[:,center].mean(1)[:,None].expand(-1,J,-1))
        value = torch.cat(parts,-1)
        assert value.shape==(B,20,172) and torch.isfinite(value).all()
        out.append(value)
    return torch.stack(out,1)
