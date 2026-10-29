"""Explicit short backend checks, separate from full hand/accuracy acceptance."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ego_hand_runtime.schema import Episode
from ego_hand_runtime.config import load_config
from ego_hand_runtime.decode import BoundedDecoder
from ego_hand_runtime.io import atomic_json
from ego_hand_runtime.timing import Trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',required=True);p.add_argument('--episode',required=True)
    p.add_argument('--component',choices=['headpose'],required=True)
    p.add_argument('--frames',type=int,default=20);p.add_argument('--output',required=True)
    args = p.parse_args()
    if args.frames<3:
        p.error('at least three frames required')
    config, episode = load_config(args.config), Episode.load(args.episode)
    import os
    from ego_hand_runtime.environment import reexec_for_config
    reexec_for_config(config['environment'])
    output = Path(args.output);output.mkdir(parents=True,exist_ok=False)
    episode.frames = episode.frames[:args.frames]
    trace = Trace(component=args.component)
    decoder = BoundedDecoder(config['decode']['window_frames'])
    from ego_hand_runtime.monitor import ResourceMonitor
    monitor=ResourceMonitor(config)
    try:
        from ego_hand_runtime.backends.headpose import StereoVO
        with trace.span('load_and_actual_headpose'):
            model = StereoVO(config)
            rows, backend = model.run(episode,decoder,trace,monitor.check)
        atomic_json(output/'head.json',rows)
        atomic_json(output/'result.json',dict(program_success=True,component=args.component,backend=backend,
                    frames=len(rows),valid=sum(r['valid'] for r in rows),GT_used=False,
                    full_pipeline_accepted=False,absolute_accuracy_evidence=False,
                    decoder_backend=decoder.backend_calls,peak_cache_frames=decoder.peak_cached))
        print('actual stereo headpose backend completed')
    except Exception as e:
        import traceback
        atomic_json(output/'failure.json',dict(error=str(e),traceback=traceback.format_exc()))
        raise
    finally:
        decoder.close();monitor.close();atomic_json(output/'resources.json',monitor.snapshot());trace.save(output/'timing.json')


if __name__=='__main__':main()
