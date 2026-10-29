"""Scoped IEEE geometry without changing the caller's visual acceleration mode.

RF-DETR 1.11.2 sets global matmul precision to 'high' on import. All Torch
backends share this lock so that one thread cannot change another's geometry.
cuVSLAM/TensorRT have independent contexts and do not use these Torch flags.
"""
from contextlib import contextmanager
from functools import wraps
import threading

SETTINGS_LOCK=threading.RLock()


def serialized_torch(fn):
    @wraps(fn)
    def run(*args,**kwargs):
        with SETTINGS_LOCK:return fn(*args,**kwargs)
    return run


@contextmanager
def geometry_precision(device):
    if str(device).split(':')[0]!='cuda':
        yield;return
    import torch
    with SETTINGS_LOCK:
        previous=torch.get_float32_matmul_precision()
        try:
            torch.set_float32_matmul_precision('highest')
            with torch.autocast(device_type='cuda',enabled=False):yield
        finally:
            torch.set_float32_matmul_precision(previous)
