"""Content-addressed assets and actual host capabilities; no automatic downloads."""
import ctypes
import importlib.util
import importlib.metadata
import os
import platform
from pathlib import Path
import shutil
import subprocess
import threading
from .errors import RuntimeFailure
from .io import sha256, identity

_DIGESTS={}
_DIGEST_LOCK=threading.Lock()


def verified_digest(path):
    """Full SHA on first use; stat-guarded reuse for pinned read-only assets."""
    path=Path(path).resolve()
    stat=path.stat()
    token=(stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,stat.st_ino,stat.st_dev)
    with _DIGEST_LOCK:
        old=_DIGESTS.get(str(path))
        if old is not None and old[0]==token:return old[1],'stat_guarded_process_cache'
        digest=sha256(path)
        after=path.stat()
        if token!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns,after.st_ino,after.st_dev):
            raise RuntimeFailure('asset_mutated','Asset changed while hashing',path=str(path))
        _DIGESTS[str(path)]=(token,digest)
        return digest,'full_SHA_this_process'


def host_resources(device='cpu'):
    import json
    result = dict(platform=platform.platform(), python=platform.python_version(), cpu_count=os.cpu_count(),
                  allowed_cpus=None, cpu_quota=None, ram_total_gib=None, ram_used_gib=None,
                  shm_available_bytes=None, gpu=None)
    if hasattr(os, 'sched_getaffinity'):
        result['allowed_cpus'] = sorted(os.sched_getaffinity(0))
    if os.name == 'nt':
        class Status(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                ['total', 'available', 'total_page', 'available_page', 'total_virtual', 'available_virtual', 'extended']]
        status = Status()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            result['ram_total_gib'] = status.total/2**30
            result['ram_used_gib'] = (status.total-status.available)/2**30
    else:
        try:
            mem = dict((line.split(':')[0], int(line.split()[1])*1024) for line in Path('/proc/meminfo').read_text().splitlines())
            result['ram_total_gib'] = mem['MemTotal']/2**30
            result['ram_used_gib'] = (mem['MemTotal']-mem.get('MemAvailable', mem['MemFree']))/2**30
            result['shm_available_bytes'] = shutil.disk_usage('/dev/shm').free
            cgroup = Path('/sys/fs/cgroup/cpu.max')
            if cgroup.exists():
                quota, period = cgroup.read_text().split()
                result['cpu_quota'] = None if quota=='max' else int(quota)/int(period)
        except (OSError, KeyError, ValueError):
            pass
    if shutil.which('nvidia-smi'):
        index = device.split(':')[-1] if device.startswith('cuda:') else '0'
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if device.startswith('cuda:') and visible is not None:
            ids = [x.strip() for x in visible.split(',') if x.strip()]
            try:
                index = ids[int(index)]
            except (IndexError, ValueError):
                result['gpu'] = dict(query_failed=True, reason='device excluded by CUDA_VISIBLE_DEVICES')
                return result
        try:
            query = subprocess.run(['nvidia-smi', '-i', index, '--query-gpu=name,driver_version,memory.total,memory.used,compute_cap',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=8, check=True)
            name, driver, total, used, compute = [x.strip() for x in query.stdout.strip().split(',')]
            result['gpu'] = dict(index=index, name=name, driver=driver, total_gib=float(total)/1024,
                                 used_gib=float(used)/1024, compute_capability=compute)
        except (subprocess.SubprocessError, ValueError):
            result['gpu'] = dict(query_failed=True)
    return result


def dependency_inventory():
    output = {}
    for module, distribution in [('numpy', 'numpy'), ('torch', 'torch'), ('cv2', 'opencv-python-headless'),
                                  ('smplx', 'smplx'), ('scipy', 'scipy'), ('h5py', 'h5py'),
                                  ('tensorrt', 'tensorrt'), ('cuvslam', 'cuvslam'), ('rfdetr', 'rfdetr'),
                                  ('ultralytics','ultralytics'),('PIL','pillow'),('pytorch_lightning','pytorch-lightning'),
                                  ('yacs','yacs'),('einops','einops'),('timm','timm'),('pyrender','pyrender'),
                                  ('safetensors','safetensors'),('dill','dill')]:
        found = importlib.util.find_spec(module)
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            version = None
        output[module] = dict(available=found is not None, version=version,
                              import_path=found.origin if found is not None else None)
    return output


def inspect_asset(name, spec, hardware=None):
    path = Path(spec.get('path', ''))
    result = dict(name=name, path=str(path), available=path.is_file(), valid=False,
                  source=spec.get('source'), expected_sha256=spec.get('sha256'))
    if not path.is_file():
        result['reason'] = 'missing'
        return result
    with path.open('rb') as f:
        is_pointer = f.read(128).startswith(b'version https://git-lfs.github.com/spec/v1')
    result.update(bytes=path.stat().st_size, LFS_pointer=is_pointer)
    if is_pointer:
        result['reason'] = 'LFS pointer is not a model'
        return result
    result['actual_sha256'],result['hash_verification']=verified_digest(path)
    if not spec.get('sha256'):
        result['reason'] = 'missing expected SHA; pin before execution'
        return result
    if spec['sha256'] != result['actual_sha256']:
        result['reason'] = 'SHA mismatch'
        return result
    if spec.get('kind') == 'trt_engine':
        expected = spec.get('compatibility', {})
        actual = hardware or {}
        if not all(k in expected for k in ['compute_capability', 'tensorrt_version', 'precision', 'model_sha256', 'shape']):
            result['reason'] = 'incomplete engine compatibility contract'
            return result
        gpu = actual.get('gpu') or {}
        if expected['compute_capability'] != gpu.get('compute_capability'):
            result['reason'] = 'GPU architecture mismatch'
            return result
        try:
            import tensorrt
            if tensorrt.__version__ != expected['tensorrt_version']:
                result['reason'] = 'TensorRT version mismatch'
                return result
        except ImportError:
            result['reason'] = 'TensorRT unavailable'
            return result
    result.update(valid=True, reason=None)
    return result


def audit(config):
    host = host_resources(config['device'])
    dependencies = dependency_inventory()
    assets = {name: inspect_asset(name, spec, host) for name, spec in config['assets'].items()}
    required = ['mano_right'] if config['backend']=='replay' else ['mano_right', 'wilor', 'wilor_config', 'mano_mean', 'side_helper', 'rf', 'lightess', 'ess_plugin']
    missing = [name for name in required if not assets.get(name, {}).get('valid')]
    if config['backend']=='live':
        missing += ['dependency:'+name for name in ['torch', 'cv2', 'smplx', 'rfdetr', 'cuvslam', 'tensorrt', 'h5py',
                    'ultralytics','PIL','pytorch_lightning','yacs','einops','timm','pyrender','dill']
                    if not dependencies[name]['available']]
        if not config['headpose']['expected_sdk_sha256']:
            missing.append('pinned actual cuVSLAM SDK SHA')
        for source in ['wilor'] + (['unidepth'] if config['depth']['mono']=='torch' else []):
            if not config['sources'].get(source) or not Path(config['sources'][source]).is_dir():
                missing.append('source:'+source)
        if config['depth']['mono']=='torch':
            missing += [n for n in ['unidepth', 'unidepth_config'] if not assets.get(n, {}).get('valid')]
            if not dependencies['safetensors']['available']:missing.append('dependency:safetensors')
        if not config['device'].startswith('cuda:'):
            missing.append('CUDA device required for live cuVSLAM/LightESS')
        elif dependencies['torch']['available']:
            import torch
            if not torch.cuda.is_available():
                missing.append('Torch CUDA runtime unavailable')
    else:
        missing += ['dependency:'+name for name in ['torch', 'smplx', 'h5py'] if not dependencies[name]['available']]
    return dict(host=host, dependencies=dependencies, assets=assets, execution_ready=not missing,
                missing=missing, custom_near_sdk_expected='ad04964ea99ba2d85bc6332c5ead1cbc4dee799c85f8501d3811a153183862fa',
                custom_near_sdk_verified=False)


def require_asset(config, name):
    spec = config['assets'].get(name, {})
    result = inspect_asset(name, spec, host_resources(config['device']) if spec.get('kind')=='trt_engine' else None)
    if not result['valid']:
        raise RuntimeFailure('asset', 'Asset missing, pointer, incompatible or hash differs', asset=name, reason=result.get('reason'))
    return Path(spec['path'])


def source_binding():
    root = Path(__file__).resolve().parent
    files = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob('*.py'))}
    return dict(files=files, sha256=identity(files))
