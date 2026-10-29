"""WiLoR-compatible MANO21. Right-canonical rotations, explicit reflection.

Only this module imports torch/smplx; planning and camera math remain lightweight.
"""
from pathlib import Path
import numpy as np
import torch
import threading
from contextlib import contextmanager
from .errors import RuntimeFailure
from .precision import geometry_precision

_PICKLE_LOCK = threading.RLock()


@contextmanager
def legacy_mano_pickle():
    """Serialize temporary NumPy/chumpy aliases across model factories."""
    import inspect
    with _PICKLE_LOCK:
        aliases = {'bool': bool, 'int': int, 'float': float, 'complex': complex,
                   'object': object, 'str': str, 'unicode': str}
        added = [k for k in aliases if k not in np.__dict__]
        old = getattr(inspect, 'getargspec', None)
        try:
            for k in added:
                setattr(np, k, aliases[k])
            if old is None:
                inspect.getargspec = inspect.getfullargspec
            yield
        finally:
            for k in added:
                delattr(np, k)
            if old is None:
                del inspect.getargspec

OPENPOSE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


def exp_so3(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    K = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(*v.shape[:-1], 3, 3)
    a = torch.linalg.vector_norm(v, dim=-1)
    return (torch.eye(3, device=v.device, dtype=v.dtype) + torch.sinc(a/torch.pi)[..., None, None]*K +
            .5*torch.sinc(a/(2*torch.pi))[..., None, None].square()*(K@K))


class NativeMano:
    def __init__(self, asset, device='cpu'):
        # Legacy licensed MANO pickles contain chumpy objects. Restrict compatibility
        # aliases to deserialization, without downgrading shared NumPy packages.
        import smplx
        from smplx.vertex_ids import vertex_ids
        # MANOLayer consumes rotation matrices directly, matching WiLoR's wrapper.
        self.device = torch.device(device)
        with legacy_mano_pickle():
            self.layer = smplx.MANOLayer(model_path=str(Path(asset)), is_rhand=True,
                                         num_betas=10, use_pca=False).to(self.device).eval()
        self.layer.requires_grad_(False)
        self.tips = torch.tensor(list(vertex_ids['mano'].values()), device=self.device, dtype=torch.long)
        self.joint_map = torch.tensor(OPENPOSE, device=self.device, dtype=torch.long)
        self.faces = np.asarray(self.layer.faces)

    def raw(self, root, local, betas):
        out = self.layer(global_orient=root[:, None], hand_pose=local, betas=betas, pose2rot=False)
        joints = torch.cat([out.joints, out.vertices[:, self.tips]], 1)[:, self.joint_map]
        return joints, out.vertices

    def tensor(self, state, betas, right, vertices=False):
        with geometry_precision(self.device),torch.autocast(device_type=self.device.type, enabled=False):
            state, betas = state.float(), betas.float()
            n = len(state)
            eye = torch.eye(3, device=self.device)[None].expand(n, 3, 3)
            local = exp_so3(state[:, 6:].reshape(n, 15, 3))
            joints, mesh = self.raw(eye, local, betas)
            sign = torch.ones(n, 3, device=self.device)
            sign[:, 0] = right.float()*2-1
            root = exp_so3(state[:, 3:6])
            xyz = torch.einsum('nij,nkj->nki', root, (joints-joints[:, :1])*sign[:, None])+state[:, None, :3]
            if not vertices:
                return xyz
            mesh = torch.einsum('nij,nkj->nki', root, (mesh-joints[:, :1])*sign[:, None])+state[:, None, :3]
            return xyz, mesh

    def __call__(self, state, betas, right, vertices=False):
        with torch.inference_mode():
            values = [torch.as_tensor(np.asarray(x), device=self.device) for x in [state, betas, right]]
            result = self.tensor(*values, vertices=vertices)
            return tuple(x.cpu().numpy() for x in result) if isinstance(result, tuple) else result.cpu().numpy()

    def export(self, state, betas, right):
        with geometry_precision(self.device),torch.inference_mode(), torch.autocast(device_type=self.device.type, enabled=False):
            state = torch.as_tensor(np.asarray(state), device=self.device, dtype=torch.float32)
            beta = torch.as_tensor(np.asarray(betas), device=self.device, dtype=torch.float32)
            side = torch.as_tensor(np.asarray(right), device=self.device)
            n = len(state)
            sign = torch.ones(n, 3, device=self.device)
            sign[:, 0] = side.float()*2-1
            S = torch.diag_embed(sign)
            root = exp_so3(state[:, 3:6])
            local = exp_so3(state[:, 6:].reshape(n, 15, 3))
            canonical_root = S@root@S
            eye = torch.eye(3, device=self.device)[None].expand(n, 3, 3)
            rest, _ = self.raw(eye, local, beta)
            # LBS rotates about the regressed wrist, not the model origin.
            transl = state[:, :3]*sign-rest[:, 0]
            reconstructed, _ = self.raw(canonical_root, local, beta)
            reconstructed = (reconstructed+transl[:, None])*sign[:, None]
            desired = self.tensor(state, beta, side)
            error = float((reconstructed-desired).abs().max()) if n else 0.
            if error>1e-6:
                raise RuntimeFailure('MANO_export', 'Exported MANO parameters do not reconstruct delivered joints', error_m=error,
                                     matmul_precision=torch.get_float32_matmul_precision())
            return {k: v.cpu().numpy() for k, v in dict(global_orient=canonical_root[:, None],
                    hand_pose=local, betas=beta, transl=transl, reflection=sign,
                    joints=reconstructed, wrist=state[:, :3]).items()}


def project_torch(points, camera):
    return project_parameters(points,torch.tensor(camera.parameters,device=points.device,dtype=torch.float32),camera.model)


def project_parameters(points,p,model):
    if p.ndim==2:p=p[:,None,:]
    if model == 'pinhole':
        xy = points[..., :2]/points[..., 2:3].clamp_min(1e-12)
    else:
        r = points[..., :2].square().sum(-1).clamp_min(1e-16).sqrt()
        theta = torch.atan2(r, points[..., 2])
        q = points[..., :2]*(theta/r)[..., None]
        r2 = q.square().sum(-1).clamp_max(torch.pi**2)
        radial, power = torch.ones_like(r2), r2
        count = 6 if model=='fisheye624' else 4
        k = p[...,4:]
        for j in range(count):
            radial = radial+k[...,j]*power
            power = power*r2
        x, y = q[..., 0]*radial, q[..., 1]*radial
        if model=='fisheye624':
            xx, yy, xy0 = x*x, y*y, x*y
            rr = xx+yy
            xy = torch.stack([x+2*k[...,7]*xy0+k[...,6]*(rr+2*xx)+k[...,8]*rr+k[...,9]*rr.square(),
                              y+2*k[...,6]*xy0+k[...,7]*(rr+2*yy)+k[...,10]*rr+k[...,11]*rr.square()], -1)
        else:
            xy = torch.stack([x, y], -1)
    return xy*p[...,:2]+p[...,2:4]
