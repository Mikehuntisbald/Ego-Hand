"""Exclusive persistent model slots and resource admission; never kill other jobs."""
from contextlib import contextmanager
import queue
import threading
import time
from .errors import RuntimeFailure
from .assets import host_resources


class ModelPool:
    def __init__(self, name, factory, capacity=1):
        self.name, self.factory = name, factory
        self.slots = queue.Queue(capacity)
        for _ in range(capacity):
            self.slots.put(None)
        self.loads = 0
        self.lock = threading.Lock()
        self.closed = False

    @contextmanager
    def acquire(self, trace, timeout=60):
        if self.closed:
            raise RuntimeFailure('pool_closed', 'Model pool has shut down', pool=self.name)
        with trace.span('model_slot_wait', pool=self.name):
            try:
                model = self.slots.get(timeout=timeout)
            except queue.Empty:
                raise RuntimeFailure('model_slot_timeout', 'Timed out waiting for a model', pool=self.name)
        try:
            if model is None:
                with trace.span('model_import_load', pool=self.name):
                    model = self.factory()
                with self.lock:
                    self.loads += 1
            yield model
        finally:
            self.slots.put(model)

    def close(self):
        self.closed = True
        while True:
            try:
                model = self.slots.get_nowait()
            except queue.Empty:
                break
            if model is not None and hasattr(model, 'close'):
                model.close()


class Admission:
    def __init__(self, config, probe=None):
        self.config, self.probe = config, probe or (lambda: host_resources(config['device']))
        self.pressured = False
        self.lock = threading.Lock()

    def check(self):
        snapshot = self.probe()
        limits = self.config['resources']
        blocked = False
        for kind, used in [('ram', snapshot.get('ram_used_gib')),
                           ('gpu', (snapshot.get('gpu') or {}).get('used_gib') if self.config['device'].startswith('cuda:') else None)]:
            if used is None:
                if kind=='gpu' and self.config['device'].startswith('cuda:'):
                    raise RuntimeFailure('resource_probe', 'GPU resource usage cannot be established')
                continue
            if used >= limits[kind+'_hard_gib']:
                self.pressured = True
                blocked = True
            elif used > limits[kind+('_resume_gib' if self.pressured else '_admit_gib')]:
                blocked = True
        self.pressured = blocked
        return not blocked, snapshot

    def wait(self, trace):
        started = time.monotonic()
        with trace.span('resource_admission'):
            while True:
                with self.lock:
                    accepted, snapshot = self.check()
                if accepted:
                    return snapshot
                if time.monotonic()-started >= self.config['scheduler']['admission_timeout_s']:
                    raise RuntimeFailure('resource_timeout', 'Resource admission timed out; no foreign process was interrupted', snapshot=snapshot)
                time.sleep(.1)
