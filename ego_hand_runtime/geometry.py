"""Physical camera and SO(3) math; no GT, learned model or CUDA imports."""
import numpy as np
from .errors import RuntimeFailure


def quaternion_matrix(q, order='xyzw'):
    q = np.asarray(q, dtype=np.float64)
    if order == 'wxyz':
        q = q[[1, 2, 3, 0]]
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise RuntimeFailure('quaternion', 'Invalid quaternion')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rotvec_matrix(v):
    v = np.asarray(v, np.float64)
    x, y, z = np.moveaxis(v, -1, 0)
    zero = np.zeros_like(x)
    K = np.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).reshape(v.shape[:-1] + (3, 3))
    a = np.linalg.norm(v, axis=-1)
    return np.eye(3) + np.sinc(a/np.pi)[..., None, None]*K + (0.5*np.sinc(a/(2*np.pi))**2)[..., None, None]*(K@K)


def matrix_rotvec(R):
    # scipy handles the pi branch and quaternion sign consistently.
    from scipy.spatial.transform import Rotation
    R = np.asarray(R)
    return Rotation.from_matrix(R.reshape(-1, 3, 3)).as_rotvec().reshape(R.shape[:-2] + (3,))


def slerp_matrix(a, b, weight):
    return np.asarray(a) @ rotvec_matrix(matrix_rotvec(np.asarray(a).swapaxes(-1, -2) @ b) * weight)


def project(points, camera):
    x = np.asarray(points, dtype=np.float64)
    p = np.asarray(camera.parameters, np.float64)
    if camera.model == 'pinhole':
        xy = x[..., :2] / np.maximum(x[..., 2:3], 1e-12)
    else:
        r = np.maximum(np.linalg.norm(x[..., :2], axis=-1), 1e-12)
        theta = np.arctan2(r, x[..., 2])
        q = x[..., :2]*(theta/r)[..., None]
        xy = distort_angle(q, camera)
    return xy*p[:2] + p[2:4]


def distort_angle(q, camera):
    p = np.asarray(camera.parameters, np.float64)
    r2 = np.minimum((q*q).sum(-1), np.pi**2)
    radial, power = np.ones_like(r2), r2.copy()
    count = 6 if camera.model == 'fisheye624' else 4
    k = p[4:]
    for index in range(count):
        radial += k[index]*power
        power *= r2
    x, y = q[..., 0]*radial, q[..., 1]*radial
    if camera.model == 'kb4':
        return np.stack([x, y], -1)
    xx, yy, xy = x*x, y*y, x*y
    rr = xx+yy
    return np.stack([x+2*k[7]*xy+k[6]*(rr+2*xx)+k[8]*rr+k[9]*rr**2,
                     y+2*k[6]*xy+k[7]*(rr+2*yy)+k[10]*rr+k[11]*rr**2], -1)


def unproject(pixels, camera):
    p = np.asarray(camera.parameters, np.float64)
    target = (np.asarray(pixels, np.float64)-p[2:4])/p[:2]
    if camera.model == 'pinhole':
        ray = np.concatenate([target, np.ones(target.shape[:-1]+(1,))], -1)
        return ray/np.linalg.norm(ray, axis=-1, keepdims=True)
    q = target.copy()
    epsilon = 1e-6
    for _ in range(20):
        value = distort_angle(q, camera)
        columns = []
        for dim in [0, 1]:
            shift = np.zeros_like(q)
            shift[..., dim] = epsilon
            columns.append((distort_angle(q+shift, camera)-value)/epsilon)
        J = np.stack(columns, -1)
        determinant = J[..., 0, 0]*J[..., 1, 1]-J[..., 0, 1]*J[..., 1, 0]
        residual = value-target
        inverse = np.where(np.abs(determinant)>1e-10, 1/np.where(determinant==0, 1, determinant), 0)
        step = np.stack([J[..., 1, 1]*residual[..., 0]-J[..., 0, 1]*residual[..., 1],
                         -J[..., 1, 0]*residual[..., 0]+J[..., 0, 0]*residual[..., 1]], -1)*inverse[..., None]
        q -= np.clip(step, -0.25, 0.25)
    angle = np.linalg.norm(q, axis=-1)
    ray = np.concatenate([q*np.sinc(angle/np.pi)[..., None], np.cos(angle)[..., None]], -1)
    error = np.linalg.norm(project(ray, camera)-pixels, axis=-1)
    if not np.isfinite(ray).all() or (error>0.05).any():
        raise RuntimeFailure('unprojection', 'Physical camera inversion failed', maximum_pixel_error=float(error.max()))
    return ray


def pixel_mask(points, uv, camera):
    return (np.isfinite(points).all(-1) & np.isfinite(uv).all(-1) & (points[..., 2] > .01) &
            (uv[..., 0] >= 0) & (uv[..., 0] < camera.width) &
            (uv[..., 1] >= 0) & (uv[..., 1] < camera.height))


def virtual_camera_map(camera, device_from_virtual, width, height, focal, principal):
    """Per-frame physical calibration onto the SAME virtual pinhole axes."""
    yy,xx=np.mgrid[:height,:width]
    rays=np.stack([(xx-principal[0])/focal,(yy-principal[1])/focal,np.ones_like(xx)],-1)
    R=camera.T_device_from_camera[:3,:3].T@device_from_virtual
    native=rays@R.T
    uv=project(native,camera);good=pixel_mask(native,uv,camera)
    uv[~good]=-1
    return uv.astype(np.float32),good


def rectified_pair(left, right, width=480, height=288, focal=200., center=None):
    a, b = left.T_device_from_camera, right.T_device_from_camera
    offset = b[:3, 3]-a[:3, 3]
    baseline = np.linalg.norm(offset)
    if baseline <= .001:
        raise RuntimeFailure('stereo_baseline', 'Physical stereo baseline is missing or degenerate')
    x = offset/baseline
    z = a[:3, 2]+b[:3, 2]
    z -= x*np.dot(x, z)
    if np.linalg.norm(z) < 1e-8:
        raise RuntimeFailure('stereo_orientation', 'No common rectification orientation')
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    device_from_rect = np.stack([x, y, np.cross(x, y)], 1)
    principal = np.array([(width-1)/2, (height-1)/2], float)
    if center is not None:
        principal -= focal*np.asarray(center)
    yy, xx = np.mgrid[:height, :width]
    rays = np.stack([(xx-principal[0])/focal, (yy-principal[1])/focal, np.ones_like(xx)], -1)
    maps, valid = [], []
    for c in [left, right]:
        R = c.T_device_from_camera[:3, :3].T @ device_from_rect
        native = rays@R.T
        uv = project(native, c)
        good = pixel_mask(native, uv, c)
        uv[~good] = -1
        maps.append(uv.astype(np.float32))
        valid.append(good)
    return dict(maps=maps, valid=valid, baseline_m=float(baseline), focal_px=float(focal),
                principal=principal, T_device_from_rect=np.block([[device_from_rect, a[:3, 3:4]],
                                                                 [np.zeros((1, 3)), np.ones((1, 1))]]))


def perspective_crop(image, box, camera, padding=1.3):
    import cv2
    box = np.asarray(box, float)
    center = (box[:2]+box[2:])/2
    z = unproject(center, camera)
    x = np.array([1., 0., 0.])-z*z[0]
    x /= np.linalg.norm(x)
    R = np.stack([x, np.cross(z, x), z], 1)
    corners = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]],
                        [center[0], box[1]], [center[0], box[3]], [box[0], center[1]], [box[2], center[1]]])
    rays = unproject(corners, camera)@R
    span = max(float(np.abs(rays[:, :2]/np.maximum(rays[:, 2:3], .05)).max()), .015)*padding
    focal = 95.5/span
    yy, xx = np.mgrid[:256, :256]
    virtual = np.stack([(xx-127.5)/focal, (yy-127.5)/focal, np.ones_like(xx)], -1)
    uv = project(virtual@R.T, camera).astype(np.float32)
    return cv2.remap(image, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT), R.astype(np.float32), focal


def surface_intersections(directions, vertices, faces):
    triangles = np.asarray(vertices)[np.asarray(faces)]
    v0 = triangles[:, 0]
    e1, e2 = triangles[:, 1]-v0, triangles[:, 2]-v0
    p = np.cross(np.asarray(directions)[:, None], e2[None])
    det = (p*e1[None]).sum(-1)
    inverse = np.divide(1., det, out=np.zeros_like(det), where=np.abs(det)>1e-9)
    u = (p*(-v0)[None]).sum(-1)*inverse
    q = np.cross(-v0, e1)
    v = (directions[:, None]*q[None]).sum(-1)*inverse
    distance = (e2*q).sum(-1)[None]*inverse
    good = (np.abs(det)>1e-9)&(u>=0)&(v>=0)&(u+v<=1)&(distance>.01)
    return np.where(good, distance, np.inf).min(-1)
