"""Bounded resource sampling and cooperative limits; never signal foreign PIDs."""
from collections import deque
import os
import threading
import time
from .assets import host_resources
from .errors import RuntimeFailure


class ResourceMonitor:
    def __init__(self, config):
        self.config = config
        self.samples = deque(maxlen=4096)
        self.stop = threading.Event()
        self.violation = None
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._run,name='ego-resource-monitor',daemon=True)
        self.thread.start()

    def _run(self):
        while not self.stop.is_set():
            began = time.perf_counter()
            row = host_resources(self.config['device'])
            row.update(timestamp_ns=time.time_ns(),PID=os.getpid(),process_cpu_s=time.process_time())
            for kind, used in [('ram',row.get('ram_used_gib')),('gpu',(row.get('gpu') or {}).get('used_gib')
                         if self.config['device'].startswith('cuda:') else None)]:
                if used is not None and used>=self.config['resources'][kind+'_hard_gib']:
                    with self.lock:
                        self.violation = dict(resource=kind,used_gib=used,limit_gib=self.config['resources'][kind+'_hard_gib'])
            row['observation_wall_s'] = time.perf_counter()-began
            with self.lock:
                self.samples.append(row)
            self.stop.wait(.5)

    def check(self):
        with self.lock:
            if self.violation:
                raise RuntimeFailure('resource_limit','Own pipeline stopped at a cooperative boundary; no foreign process signaled',**self.violation)

    def snapshot(self):
        with self.lock:
            return dict(samples=list(self.samples),violation=self.violation,sampling_interval_s=.5,
                        accounting='host/device totals; process_cpu is cumulative, not additive across spans',
                        hard_limit_semantics='cooperative boundary guard, not a CUDA allocator/cgroup hard cap')

    def close(self):
        self.stop.set();self.thread.join(timeout=10)
