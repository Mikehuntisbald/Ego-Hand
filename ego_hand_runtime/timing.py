"""Nested wall/thread/process CPU spans. Nested CPU totals are not additive."""
import contextvars
from contextlib import contextmanager
import threading
import time
import uuid
from .io import atomic_json

_PARENT = contextvars.ContextVar('ego_span_parent', default=None)


class Trace:
    def __init__(self, **ids):
        self.ids = ids
        self.events, self.lock = [], threading.Lock()
        self.start_wall = time.perf_counter()
        self.start_cpu = time.process_time()

    @contextmanager
    def span(self, name, **fields):
        event_id = uuid.uuid4().hex
        parent = _PARENT.get()
        token = _PARENT.set(event_id)
        start, thread, process = time.perf_counter(), time.thread_time(), time.process_time()
        success = False
        try:
            yield
            success = True
        finally:
            event = dict(name=name, span_id=event_id, parent_id=parent,
                         relative_start_s=start-self.start_wall, wall_s=time.perf_counter()-start,
                         thread_cpu_s=time.thread_time()-thread, process_cpu_s=time.process_time()-process,
                         success=success, thread_id=threading.get_ident(), **self.ids, **fields)
            _PARENT.reset(token)
            with self.lock:
                self.events.append(event)

    def save(self, path):
        atomic_json(path, dict(schema=1, wall_s=time.perf_counter()-self.start_wall,
                              process_cpu_s=time.process_time()-self.start_cpu,
                              cpu_spans_additive=False, observation_overhead_included=True,
                              events=self.events))
