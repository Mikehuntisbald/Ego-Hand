"""Loopback-only bounded episode service; one set of persistent model pools."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import threading
import time
import uuid
from .pipeline import RuntimeContext, infer
from .io import atomic_json, jsonable
from .errors import RuntimeFailure


class Scheduler:
    def __init__(self, config, receipt_dir):
        self.config, self.context = config, RuntimeContext(config)
        self.receipt_dir = Path(receipt_dir)
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(config['scheduler']['queue_capacity'])
        self.jobs, self.lock, self.stopping = {}, threading.RLock(), False
        self.workers = [threading.Thread(target=self._worker, name='ego-worker-'+str(i))
                        for i in range(config['scheduler']['workers'])]
        for worker in self.workers:
            worker.start()
        self.started = time.perf_counter()

    def _save(self, key, **fields):
        with self.lock:
            self.jobs[key].update(fields)
            atomic_json(self.receipt_dir/(key+'.json'), self.jobs[key])

    def submit(self, request):
        allowed = {'episode', 'output', 'start_ns', 'end_ns', 'resume', 'rerun'}
        if set(request)-allowed or not {'episode', 'output'}<=set(request):
            raise RuntimeFailure('request', 'Request requires episode/output and only inference arguments')
        for field in ['resume', 'rerun']:
            if field in request and not isinstance(request[field], bool):
                raise RuntimeFailure('request', 'Resume/rerun must be boolean')
        # Resolve before queueing. No worker receives relative cwd-dependent paths.
        request = dict(request, episode=str(Path(request['episode']).resolve()), output=str(Path(request['output']).resolve()))
        key = uuid.uuid4().hex
        with self.lock:
            if self.stopping:
                raise RuntimeFailure('stopping', 'Service is draining its own queued work')
            self.jobs[key] = dict(request_id=key, status='queued', submitted_ns=time.time_ns(), request=request)
            try:
                self.queue.put_nowait((key, request))
            except queue.Full:
                del self.jobs[key]
                raise RuntimeFailure('queue_full', 'Bounded episode queue is full; retry later')
            self._save(key)
        return dict(request_id=key, status='queued')

    def _worker(self):
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                key, request = item
                self._save(key, status='running', started_ns=time.time_ns(), worker_id=threading.get_ident())
                try:
                    result = infer(request['episode'], self.config, request['output'],
                                   context=self.context, request_id=key,
                                   **{k:v for k,v in request.items() if k not in {'episode','output'}})
                    self._save(key, status='succeeded', result=result, completed_ns=time.time_ns())
                except Exception as exc:
                    error = exc.as_dict() if isinstance(exc, RuntimeFailure) else dict(code=type(exc).__name__, message=str(exc))
                    self._save(key, status='failed', error=error, completed_ns=time.time_ns())
            finally:
                self.queue.task_done()

    def snapshot(self, key=None):
        with self.lock:
            if key:
                if key not in self.jobs:
                    raise RuntimeFailure('request_id', 'Unknown request')
                return dict(self.jobs[key])
            return dict(jobs={k:v['status'] for k,v in self.jobs.items()}, queued=self.queue.qsize(),
                        pool_loads=self.context.counts(), stopping=self.stopping)

    def close(self):
        with self.lock:
            self.stopping = True
        self.queue.join()
        for _ in self.workers:
            self.queue.put(None)
        for worker in self.workers:
            worker.join()
        self.context.close()
        atomic_json(self.receipt_dir/'resources_tail.json',self.context.monitor.snapshot())
        atomic_json(self.receipt_dir/'service_exit.json', dict(wall_through_shutdown_s=time.perf_counter()-self.started,
                    snapshot=self.snapshot(), includes_model_load_and_service_shutdown=True,
                    performance_admitted=False, nested_CPU_spans_additive=False))


def serve(config, host, port, receipts):
    if host!='127.0.0.1':
        raise RuntimeFailure('bind', 'This local runtime accepts loopback only')
    scheduler = Scheduler(config, receipts)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def respond(self, code, obj):
            data = json.dumps(jsonable(obj), allow_nan=False).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            try:
                self.respond(200, scheduler.snapshot(self.path.removeprefix('/jobs/')) if self.path.startswith('/jobs/') else scheduler.snapshot())
            except RuntimeFailure as e:
                self.respond(404, e.as_dict())
        def do_POST(self):
            try:
                n = int(self.headers.get('Content-Length', '0'))
                if not 0<n<65536:
                    raise RuntimeFailure('request_size', 'Invalid request length')
                data = json.loads(self.rfile.read(n))
                if self.path=='/submit':
                    self.respond(202, scheduler.submit(data))
                elif self.path=='/shutdown' and data=={'drain': True}:
                    with scheduler.lock:
                        scheduler.stopping = True
                    self.respond(202, dict(draining=True))
                    threading.Thread(target=server.shutdown, daemon=True).start()
                else:
                    raise RuntimeFailure('request', 'Unsupported operation')
            except (RuntimeFailure, ValueError, TypeError) as e:
                self.respond(400, e.as_dict() if isinstance(e, RuntimeFailure) else dict(error=str(e)))
    server = ThreadingHTTPServer((host, port), Handler)
    try:
        print(json.dumps(dict(listening='http://'+host+':'+str(server.server_port), PID=__import__('os').getpid())), flush=True)
        server.serve_forever(poll_interval=.2)
    finally:
        server.server_close()
        scheduler.close()
