"""General episode orchestration. Frozen research entry points are never runpy'd."""
from contextlib import contextmanager
from pathlib import Path
import copy
import os
import threading
import time
import uuid
import numpy as np
from .assets import audit, require_asset, source_binding
from .errors import RuntimeFailure
from .io import atomic_json, read_json, sha256, identity, seal, verify_seal
from .schema import Episode, reject_privileged
from .timing import Trace
from .pool import ModelPool, Admission


def pid_alive(pid):
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            # Access denied means live/unknown, never silently steal its lock.
            return ctypes.get_last_error()!=87
        try:
            status = wintypes.DWORD()
            return not kernel.GetExitCodeProcess(handle, ctypes.byref(status)) or status.value==259
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def process_identity(pid):
    """A reused PID/another container's PID 1 must not inherit an old lock."""
    if os.name!='nt':
        try:
            stat=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
            return dict(start_ticks=stat[19],pid_namespace=os.readlink('/proc/'+str(pid)+'/ns/pid'),
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        except (OSError,IndexError):return None
    import ctypes
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.OpenProcess.restype=wintypes.HANDLE
    kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
    kernel.CloseHandle.argtypes=[wintypes.HANDLE]
    kernel.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    handle=kernel.OpenProcess(0x1000,False,int(pid))
    if not handle:return None
    try:
        times=[wintypes.FILETIME() for _ in range(4)]
        if not kernel.GetProcessTimes(handle,*[ctypes.byref(t) for t in times]):return None
        return dict(start_filetime=(times[0].dwHighDateTime<<32)|times[0].dwLowDateTime)
    finally:kernel.CloseHandle(handle)


@contextmanager
def output_lock(root):
    path = root / 'run.lock'
    # Kernel locks survive neither SIGKILL nor container exit. Keep the inode
    # instead of unlinking it, which could let two owners lock different inodes.
    stream=path.open('a+b');acquired=False;owner=None
    kind='msvcrt' if os.name=='nt' else 'flock'
    try:
        if path.stat().st_size==0:stream.write(b' ');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            acquired=True
        except OSError:
            raise RuntimeFailure('output_busy','Another process holds the output kernel lock')
        stream.seek(0);raw=stream.read().strip()
        import json
        old=json.loads(raw) if raw else None
        if old and not old.get('released'):
            if old.get('kernel_lock') and old['kernel_lock']!=kind:
                raise RuntimeFailure('output_busy','Unreleased lock belongs to another OS backend; use the same backend or isolated rerun')
            if not old.get('kernel_lock'):
                current=process_identity(old['pid'])
                namespace_unknown=(old.get('process_identity') or {}).get('pid_namespace')!=(current or {}).get('pid_namespace')
                if pid_alive(old['pid']) and (current is None or namespace_unknown or old.get('process_identity') is None or current==old['process_identity']):
                    raise RuntimeFailure('output_busy','Legacy live/unknown owner must not be stolen',pid=old['pid'])
            atomic_json(root/('stale-lock-'+uuid.uuid4().hex+'.json'),old)
        owner=dict(pid=os.getpid(),process_identity=process_identity(os.getpid()),time_ns=time.time_ns(),kernel_lock=kind,released=False)
        def record(value):
            stream.seek(0);stream.truncate();stream.write(json.dumps(value).encode());stream.flush();os.fsync(stream.fileno())
        record(owner)
        yield
    finally:
        if owner is not None:
            owner.update(released=True,released_ns=time.time_ns());record(owner)
        if acquired:
            stream.seek(0)
            if os.name=='nt':msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(stream.fileno(),fcntl.LOCK_UN)
        stream.close()


def plan(episode, config, start_ns=None, end_ns=None, *, preflight=True):
    from .clock import sample_indices
    interval = episode.interval(start_ns, end_ns)
    if not interval:
        raise RuntimeFailure('interval', 'Requested interval contains no source frames')
    sampled = [interval[i] for i in sample_indices(episode.source_clock_ns[interval], config['sampling']['primary_fps'])]
    capabilities = episode.capabilities()
    report = audit(config) if preflight else None
    missing = list(report['missing']) if report else []
    if config['backend']=='live' and not capabilities['stereo_vo']:
        missing.append('real stereo tracking cameras')
    if config['backend']=='replay' and not Path(config['replay'].get('bundle') or '').is_dir():
        missing.append('sealed prediction replay bundle')
    return dict(schema=1, episode_id=episode.episode_id, source_id=episode.source_id,
                episode_manifest=str(episode.source_path),episode_manifest_sha256=sha256(episode.source_path),
                source_frames=len(episode.frames), interval_indices=interval, inference_indices=sampled,
                output_frames=len(episode.frames), capabilities=capabilities, backend=config['backend'],
                execution_ready=not missing, missing=missing, preflight=report,
                DiT_enabled=False, headpose='pure stereo VO; no IMU fusion',
                side_recovery_enabled=False, production_parity=False, GT_used=False)


def binding_for(episode, config, report, start_ns, end_ns):
    # Include actual input bytes, not just their names/mtime. Missing frames are
    # explicit invalidity; never replace them with generated image evidence.
    image_files = sorted({im['path'] for f in episode.frames for im in f['images'].values()})
    inputs = {name: sha256(name) if Path(name).is_file() else None for name in image_files}
    sources = {}
    for name, folder in config['sources'].items():
        base = Path(folder)
        sources[name] = {p.relative_to(base).as_posix(): sha256(p) for p in sorted(base.rglob('*.py'))}
    replay = config['replay'].get('bundle')
    if config['backend']=='replay' and replay:
        inputs['replay_seal'] = sha256(Path(replay)/'inference_seal.json')
    return identity(dict(episode_sha=sha256(episode.source_path), inputs=inputs, config=config,
                         assets={k: v.get('actual_sha256') for k, v in report['assets'].items()},
                         runtime=source_binding()['sha256'], dependency_sources=sources,
                         interval=[start_ns, end_ns]))


class RuntimeContext:
    """One owner per process, reused by all episode workers. Slot leases are exclusive."""
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.source_sha256 = source_binding()['sha256']
        self.admission = Admission(config)
        from .monitor import ResourceMonitor
        self.monitor = ResourceMonitor(config)
        self.pools = {}
        def make(name):
            if name=='ba':
                from .native import NativeMano
                return NativeMano(require_asset(config, 'mano_right'), config['device'])
            if name=='rf':
                from .backends.vision import Detector
                return Detector(config)
            if name=='wilor':
                from .backends.vision import WiLoR
                return WiLoR(config)
            if name=='headpose':
                from .backends.headpose import StereoVO
                return StereoVO(config)
            from .backends.stereo import LightESS, UniDepth
            return LightESS(config) if name=='stereo' else UniDepth(config)
        # No lazy import/model construction is attempted during a dry run.
        for name, count in config['pools'].items():
            self.pools[name] = ModelPool(name, lambda name=name: make(name), count)

    def close(self):
        for pool in self.pools.values():
            pool.close()
        self.monitor.close()

    def counts(self):
        return {k: p.loads for k, p in self.pools.items()}


def _stage(root, name, fn, resume, trace):
    path = root/(name+'.json')
    receipt = root/(name+'.receipt.json')
    if resume and receipt.is_file():
        if not path.is_file() or sha256(path)!=read_json(receipt)['sha256']:
            raise RuntimeFailure('stage_corrupt', 'Saved stage hash differs; preserve output and rerun in a new directory', stage=name)
        with trace.span('stage_resume_verified', stage=name):
            return read_json(path)
    with trace.span(name):
        value = fn()
        atomic_json(path, value)
        atomic_json(receipt, dict(sha256=sha256(path)))
        return value


def _replay(config):
    root = Path(config['replay']['bundle'])
    receipt = verify_seal(root)
    for file in ['head.json', 'observations.json','provenance.json']:
        if file not in receipt['files']:
            raise RuntimeFailure('replay_seal', 'Replay evidence is not covered by seal', file=file)
    head, rows = read_json(root/'head.json'), read_json(root/'observations.json')
    reject_privileged(rows)
    reject_privileged(head)
    if read_json(root/'provenance.json').get('kind') not in {'recorded_predictions', 'synthetic_contract_fixture'}:
        raise RuntimeFailure('replay_provenance', 'Replay must be explicit recorded predictions or a named contract fixture')
    return dict(head=head, observations=rows, backend='sealed_recorded_predictions_not_live_GPU',
                provenance=read_json(root/'provenance.json'))


def _depth_fill(rows):
    """Same-track sparse anchors, bounded by real time. No unbounded edge hold."""
    for row in rows:
        if row.get('depth_valid'):
            continue
        known = sorted([r for r in rows if r['track_id']==row['track_id'] and r['right']==row['right']
                        and r.get('depth_valid') and not r.get('depth_interpolated')], key=lambda x:x['timestamp_ns'])
        if not known:
            continue
        t = np.array([r['timestamp_ns'] for r in known], np.int64)
        stamp = row['timestamp_ns']
        j = int(np.argmin(np.abs(t-stamp)))
        if abs(int(t[j])-stamp)>71000000:
            continue
        k = int(np.searchsorted(t, stamp))
        depth = known[j]['depth_m']
        if 0<k<len(t) and int(t[k]-t[k-1])<100000000:
            w = (stamp-int(t[k-1]))/int(t[k]-t[k-1])
            depth = known[k-1]['depth_m']*(1-w)+known[k]['depth_m']*w
        row.update(depth_m=depth, depth_valid=True, depth_interpolated=True,
                   depth_source='time_bounded:'+known[j]['depth_source'], depth_weight=known[j].get('depth_weight', 1.))


def _live(episode, interval, sampled, config, context, root, resume, trace):
    from .decode import BoundedDecoder
    from .backends.vision import associate_and_vote
    from .backends.stereo import estimate, mono_anchor
    primary = episode.camera_for_role('rgb')
    decoder = BoundedDecoder(config['decode']['window_frames'])
    working = copy.copy(episode)
    working.frames = [episode.frames[i] for i in interval]
    try:
        def run_head():
            with context.pools['headpose'].acquire(trace) as model:
                frames, backend = model.run(working, decoder, trace, context.monitor.check)
                return dict(frames=frames, backend=backend)
        head = _stage(root, 'head', run_head, resume, trace)

        def detect():
            rows = []
            (root/'masks').mkdir(exist_ok=True)
            with context.pools['rf'].acquire(trace) as model:
                for index in sampled:
                    frame = episode.frames[index]
                    context.monitor.check()
                    im = frame['images'].get(primary.id)
                    if not im or not im['valid']:
                        continue
                    with trace.span('primary_decode'):
                        image = decoder.read(im)
                    with trace.span('RF_model'):
                        found, masks = model.predict(image)
                    for j, row in enumerate(found):
                        row.update(timestamp_ns=frame['timestamp_ns'], source_timestamp_ns=im['timestamp_ns'],
                                   frame_index=index, source_index=im['source_index'], camera_id=primary.id)
                        if masks is not None:
                            name = 'masks/'+str(index)+'-'+str(j)+'.npz'
                            np.savez_compressed(root/name, mask=masks[j])
                            row['mask_file'] = name
                            row['mask_sha256'] = sha256(root/name)
                        rows.append(row)
            return associate_and_vote(rows,config['quality']['min_side_confidence'])
        detections = _stage(root, 'detections', detect, resume, trace)
        def predict():
            observations = []
            mono_frame,mono_depth=None,None
            # Lease MANO before WiLoR consistently across workers (no lock inversion).
            with context.pools['ba'].acquire(trace) as codec, context.pools['wilor'].acquire(trace) as model:
                for row in detections:
                    context.monitor.check()
                    frame = episode.frames[row['frame_index']]
                    if mono_frame!=row['frame_index']:
                        mono_frame,mono_depth=row['frame_index'],None
                    actual_camera = episode.camera_at(primary.id, row['timestamp_ns'])
                    with trace.span('wilor_decode'):
                        image = decoder.read(frame['images'][primary.id])
                    with trace.span('WiLoR_model_geometry'):
                        output, vertices, joints = model.predict(image, row, actual_camera, codec)
                    output.update(depth_valid=False, depth_m=None, depth_source='not_sampled')
                    mask = None
                    if row.get('mask_file'):
                        if sha256(root/row['mask_file'])!=row['mask_sha256']:
                            raise RuntimeFailure('mask_resume','Saved instance-mask bytes changed')
                        with np.load(root/row['mask_file']) as data:
                            mask = data['mask']
                    if row['frame_index'] in sampled[::config['sampling']['stereo_stride']]:
                        with context.pools['stereo'].acquire(trace) as engine:
                            output.update(estimate(output, vertices, joints, codec.faces, mask,
                                                   episode, frame, decoder, engine, config, trace))
                        if not output['depth_valid'] and config['depth']['mono']=='torch':
                            if mono_depth is None:
                                with context.pools['unidepth'].acquire(trace) as mono:
                                    with trace.span('mono_model'):
                                        mono_depth = mono(image, actual_camera)
                            else:
                                with trace.span('mono_same_frame_cache_hit'):pass
                            output.update(mono_anchor(mono_depth, mask, vertices, joints, codec.faces, actual_camera, config))
                    observations.append(output)
            # Reject mono disagreement with adjacent reliable physical anchors.
            for row in observations:
                if row.get('depth_valid') and row.get('depth_weight')==.25:
                    stereo = [x for x in observations if x['track_id']==row['track_id'] and x.get('depth_valid')
                              and x.get('depth_weight')==1. and abs(x['timestamp_ns']-row['timestamp_ns'])<=200000000]
                    if stereo and min(abs(x['depth_m']-row['depth_m']) for x in stereo)>max(.05, .25*row['depth_m']):
                        row.update(depth_valid=False, depth_m=None, depth_source='mono_stereo_conflict')
            _depth_fill(observations)
            return observations
        rows = _stage(root, 'observations', predict, resume, trace)
        return dict(head=head['frames'], observations=rows, backend=head['backend'],
                    decoder=dict(actual_backends=decoder.backend_calls, peak_cache_frames=decoder.peak_cached,
                                 configured_window_frames=decoder.limit))
    finally:
        decoder.close()


def infer(episode_path, config, output, *, start_ns=None, end_ns=None, dry_run=False,
          resume=False, rerun=False, context=None, request_id=None):
    from .observations import route
    episode = Episode.load(episode_path)
    started = time.perf_counter()
    trace = Trace(episode_id=episode.episode_id, source_id=episode.source_id,
                  request_id=request_id or uuid.uuid4().hex, worker_id=str(threading.get_ident()))
    with trace.span('manifest_preflight'):
        planned = plan(episode, config, start_ns, end_ns)
    if dry_run:
        return planned
    root = Path(output).resolve()
    if resume and rerun:
        raise RuntimeFailure('resume', 'Choose resume or isolated rerun')
    if rerun and root.exists():
        root = root/('attempt-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    if root.exists() and not resume:
        raise RuntimeFailure('output_exists', 'Output exists; use --resume or --rerun')
    root.mkdir(parents=True, exist_ok=True)
    own_context = context is None
    with output_lock(root):
        try:
            if not planned['execution_ready']:
                atomic_json(root/'preflight_failed.json',planned)
                raise RuntimeFailure('preflight', 'Required execution assets/capabilities are missing', missing=planned['missing'])
            with trace.span('content_hash_preflight'):
                binding = binding_for(episode, config, planned['preflight'], start_ns, end_ns)
            binding_file = root/'binding.json'
            if binding_file.is_file() and read_json(binding_file)['sha256']!=binding:
                raise RuntimeFailure('resume_mismatch', 'Source/input/assets/config changed; use an isolated rerun')
            atomic_json(root/'plan.json',planned)
            atomic_json(binding_file, dict(sha256=binding))
            if resume and (root/'inference_seal.json').is_file():
                verify_seal(root, binding)
                receipt = read_json(root/'result.json')
                return dict(receipt, resumed_complete=True)
            atomic_json(root/'expanded_config.json', config)
            atomic_json(root/'input_contract.json',read_json(episode.source_path))
            if context is None:
                context = RuntimeContext(config)
            elif context.config!=config or context.source_sha256!=source_binding()['sha256']:
                raise RuntimeFailure('service_source_changed','Running service config/source differs; drain and restart before continuing')
            admitted = context.admission.wait(trace)
            context.monitor.check()
            atomic_json(root/'admission.json', admitted)
            evidence = _stage(root, 'evidence', lambda: _replay(config) if config['backend']=='replay' else
                              _live(episode, planned['interval_indices'], planned['inference_indices'], config,
                                    context, root, resume, trace), resume, trace)
            selected_stamps = set(int(episode.source_clock_ns[i]) for i in planned['inference_indices'])
            rows = [r for r in evidence['observations'] if r['timestamp_ns'] in selected_stamps]
            # A requested interval must not use outside head poses to invent
            # valid delivery outside that interval.
            start = int(episode.source_clock_ns[planned['interval_indices'][0]])
            end = int(episode.source_clock_ns[planned['interval_indices'][-1]])
            head = evidence['head']
            seeds, observations, rejected = route(rows, head, episode, config['ba']['max_gap_ns'])
            invalid_times = {}
            for row in [r for r in rows if not r.get('valid',True)]+rejected:
                invalid_times.setdefault(str(row['track_id']),set()).add(row['timestamp_ns'])
            atomic_json(root/'route.json', dict(states=len(seeds), observations=len(observations), rejected=rejected))
            from .ba import optimize
            from .quality import protect
            from .delivery import write, verify
            with context.pools['ba'].acquire(trace) as codec:
                context.monitor.check()
                with trace.span('native_MANO_BA_control'):
                    baseline = optimize(seeds, observations, codec, config['ba'], disable_additions=True,guard=context.monitor.check)
                additions = config['ba']['mask_outside_image'] or any(config['ba'][k] for k in
                    ['world_root_so3_weight', 'wrist_motion_weight', 'observation_preservation_weight'])
                with trace.span('native_MANO_BA_proposal'):
                    proposal = optimize(seeds, observations, codec, config['ba'],guard=context.monitor.check) if additions else baseline
                with trace.span('quality_parameter_protection'):
                    result, protection = protect(baseline, proposal, seeds, observations, codec, config['quality'])
                atomic_json(root/'optimization.json', dict(control=baseline, proposal=proposal, protection=protection,
                                                          no_additions_exact_control=not additions))
                with trace.span('source_clock_write_FK_verify'):
                    delivered = write(episode, seeds, result, head, codec, root, config, invalid_times,(start,end))
                    check = verify(root)
                from .qa import summary
                atomic_json(root/'qa.json',summary(delivered,config['ba']['max_gap_ns']))
            receipt = dict(success=True, program_success=True, quality_admitted=False,
                           QA='geometry/contracts only; no independent accuracy evaluation',
                           performance_admitted=False, backend=config['backend'], production_parity=False,
                           output=str(root), source_frames=len(episode.frames), states=len(seeds),
                           input_observations=len(rows),routed_observations=len(observations),
                           depth_supported=sum(bool(r.get('depth_valid')) for r in rows),
                           depth_sources={s:sum(bool(r.get('depth_valid')) and r.get('depth_source')==s for r in rows)
                                          for s in sorted({r.get('depth_source','unavailable') for r in rows})},
                           delivery=check, pool_loads=context.counts(), GT_used=False,
                           inference_wall_s=time.perf_counter()-started)
            atomic_json(root/'result.json', receipt)
            atomic_json(root/'resources.json', context.monitor.snapshot())
            files = ['joints.json', 'bundle.h5', 'evidence.json', 'optimization.json', 'result.json',
                     'route.json', 'expanded_config.json', 'binding.json', 'resources.json','qa.json','input_contract.json']
            with trace.span('seal_outputs'):
                seal(root, files, binding=binding, backend=config['backend'])
            return receipt
        except Exception as exc:
            error = exc.as_dict() if isinstance(exc, RuntimeFailure) else dict(code=type(exc).__name__, message=str(exc))
            import traceback
            name = 'failure-'+str(time.time_ns())+'.json'
            atomic_json(root/name, dict(success=False, error=error, traceback=traceback.format_exc(), PID=os.getpid()))
            raise
        finally:
            trace.save(root/('timing-'+str(time.time_ns())+'.json'))
            if own_context and context is not None:
                context.close()
