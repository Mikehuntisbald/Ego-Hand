"""Incremental environment restoration and exact Kubernetes service-link groups."""
import os
import re
import threading
from contextlib import contextmanager

_LOCK = threading.RLock()
_MISSING = object()


def service_link_keys(environ):
    remove = set()
    for key, host in environ.items():
        if not key.endswith('_SERVICE_HOST'):
            continue
        prefix = key.removesuffix('_SERVICE_HOST')
        port = environ.get(prefix+'_SERVICE_PORT', '')
        if not port.isdigit() or not host:
            continue
        uri = environ.get(prefix+'_PORT', '')
        host_uri = '['+host+']' if ':' in host else host
        if uri != 'tcp://'+host_uri+':'+port:
            continue
        generated = prefix+'_PORT_'+port+'_TCP'
        required = {generated: uri, generated+'_PROTO': 'tcp', generated+'_PORT': port,
                    generated+'_ADDR': host}
        if not all(environ.get(k)==v for k, v in required.items()):
            continue
        group = {key, prefix+'_SERVICE_PORT', prefix+'_PORT', *required}
        for candidate, value in environ.items():
            if candidate.startswith(prefix+'_SERVICE_PORT_') and value.isdigit():
                group.add(candidate)
            if re.fullmatch(re.escape(prefix)+r'_PORT_[0-9]+_TCP(?:_(?:PROTO|PORT|ADDR))?', candidate):
                # Only complete port subgroups are removed, below.
                sub = candidate.split('_TCP')[0]+'_TCP'
                number = sub.removeprefix(prefix+'_PORT_').removesuffix('_TCP')
                expected_uri = 'tcp://'+host_uri+':'+number
                expected = {sub: expected_uri, sub+'_PROTO': 'tcp', sub+'_PORT': number, sub+'_ADDR': host}
                if all(environ.get(k)==v for k, v in expected.items()):
                    group.update(expected)
        remove.update(group)
    return remove


def child_environment(overrides, inherited=None):
    inherited = dict(os.environ if inherited is None else inherited)
    filtered = service_link_keys(inherited)
    # Legacy HAMR4D flags are never implicitly trusted. Explicit config overrides
    # are applied after cleanup and recorded in the request receipt.
    filtered.update(k for k in inherited if k.startswith('HAMR4D_'))
    result = {k: v for k, v in inherited.items() if k not in filtered}
    result.update({str(k): str(v) for k, v in overrides.items()})
    return result


def reexec_for_config(overrides, argv=None):
    """Apply config BEFORE native-library initialization (LD search is cached).

    One process-boundary replacement, never per-worker environment mutation.
    """
    import sys
    effective=child_environment(overrides,os.environ)
    if effective!=dict(os.environ):
        os.execve(sys.executable,argv or [sys.executable,*sys.argv],effective)


@contextmanager
def environment_delta(overrides, remove=()):
    """For isolated subprocess setup/tests, not concurrent model-call mutation."""
    with _LOCK:
        changed = {k: os.environ.get(k, _MISSING) for k in set(overrides)|set(remove)
                   if k in remove or os.environ.get(k, _MISSING)!=str(overrides[k])}
        try:
            for key in remove:
                os.environ.pop(key, None)
            for key, value in overrides.items():
                if key in changed:
                    os.environ[key] = str(value)
            yield len(changed)
        finally:
            for key, value in changed.items():
                if value is _MISSING:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
