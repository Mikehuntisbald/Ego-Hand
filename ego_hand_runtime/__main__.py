"""python -m ego_hand_runtime; no model imports for help/planning."""
import argparse
import json
import os
from pathlib import Path
import sys
from .errors import RuntimeFailure
from .io import atomic_json, jsonable, verify_seal


def main(argv=None):
    parser = argparse.ArgumentParser(description='Ego-Hand independent stereo/native-MANO runtime')
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare-hot3d')
    prep.add_argument('--input', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--episode-id')
    for name in ['doctor','infer','serve']:
        cmd = commands.add_parser(name)
        cmd.add_argument('--config')
        cmd.add_argument('--device')
        if name=='doctor':
            cmd.add_argument('--output')
        elif name=='serve':
            cmd.add_argument('--host', default='127.0.0.1')
            cmd.add_argument('--port', type=int, default=8765)
            cmd.add_argument('--receipts', required=True)
        else:
            cmd.add_argument('--episode', required=True)
            cmd.add_argument('--output', required=True)
            cmd.add_argument('--start-ns', type=int)
            cmd.add_argument('--end-ns', type=int)
            cmd.add_argument('--dry-run', action='store_true')
            cmd.add_argument('--resume', action='store_true')
            cmd.add_argument('--rerun', action='store_true')
    verify = commands.add_parser('verify')
    verify.add_argument('--output', required=True)
    score = commands.add_parser('score')
    score.add_argument('--predictions', required=True)
    score.add_argument('--labels', required=True)
    score.add_argument('--output', required=True)
    submit = commands.add_parser('submit')
    submit.add_argument('--url', default='http://127.0.0.1:8765')
    submit.add_argument('--episode', required=True)
    submit.add_argument('--output', required=True)
    submit.add_argument('--wait', action='store_true')
    submit.add_argument('--resume', action='store_true')
    submit.add_argument('--start-ns',type=int)
    submit.add_argument('--end-ns',type=int)
    status = commands.add_parser('status')
    status.add_argument('--url', default='http://127.0.0.1:8765')
    status.add_argument('--request-id')
    stop = commands.add_parser('stop-service')
    stop.add_argument('--url', default='http://127.0.0.1:8765')
    args = parser.parse_args(argv)
    try:
        if args.command in {'doctor','infer','serve'}:
            from .config import load_config
            config = load_config(args.config, args.device)
            # Apply once at this process boundary. No per-worker clear/update of
            # the global environment, and no inherited HAMR4D switch bypasses config.
            from .environment import reexec_for_config
            reexec_for_config(config['environment'],[sys.executable,'-m','ego_hand_runtime',*(sys.argv[1:] if argv is None else argv)])
        if args.command=='prepare-hot3d':
            from .adapters.hot3d import prepare
            result = prepare(args.input,args.output,args.episode_id)
            result = dict(manifest=str(Path(args.output).resolve()), frames=len(result['frames']))
        elif args.command=='doctor':
            from .assets import audit
            result = audit(config)
            if args.output:
                atomic_json(args.output,result)
        elif args.command=='infer':
            from .pipeline import infer
            result = infer(args.episode,config,args.output,start_ns=args.start_ns,end_ns=args.end_ns,
                           dry_run=args.dry_run,resume=args.resume,rerun=args.rerun)
        elif args.command=='verify':
            from .delivery import verify
            verify_seal(args.output)
            result = verify(args.output)
        elif args.command=='serve':
            from .service import serve
            serve(config,args.host,args.port,args.receipts)
            return 0
        elif args.command=='score':
            from .scoring import score
            result = score(args.predictions,args.labels,args.output)
        else:
            import time
            import urllib.request
            import urllib.parse
            parsed = urllib.parse.urlparse(args.url)
            if parsed.scheme!='http' or parsed.hostname!='127.0.0.1':
                raise RuntimeFailure('endpoint','This client only contacts the local loopback service')
            def call(path,data=None):
                request = urllib.request.Request(args.url.rstrip('/')+path,
                    data=json.dumps(data).encode() if data is not None else None,
                    headers={'Content-Type':'application/json'})
                # A localhost job never needs the user's external HTTP proxy.
                with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=20) as response:
                    return json.load(response)
            if args.command=='submit':
                result = call('/submit',dict(episode=str(Path(args.episode).resolve()),output=str(Path(args.output).resolve()),resume=args.resume,
                                            start_ns=args.start_ns,end_ns=args.end_ns))
                if args.wait:
                    while result['status'] in {'queued','running'}:
                        time.sleep(.2)
                        result = call('/jobs/'+result['request_id'])
                    if result['status']=='failed':
                        print(json.dumps(result));return 1
            elif args.command=='stop-service':
                result = call('/shutdown',dict(drain=True))
            else:
                result = call('/jobs/'+args.request_id if args.request_id else '/')
        print(json.dumps(jsonable(result), ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as exc:
        error = exc.as_dict() if isinstance(exc, RuntimeFailure) else dict(code=type(exc).__name__,message=str(exc))
        print(json.dumps(dict(success=False,error=error),ensure_ascii=False),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
