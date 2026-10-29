"""Build a hardware-specific LightESS engine from SHA-pinned public ONNX/plugin.

Never deserialize an engine built for a different GPU to 'convert' it.
"""
import argparse
import ctypes
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ego_hand_runtime.io import sha256,atomic_json
from ego_hand_runtime.assets import host_resources


def main():
    p=argparse.ArgumentParser()
    for name in ['onnx','onnx-sha256','plugin','plugin-sha256','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda:0');p.add_argument('--workspace-gib',type=float,default=2.)
    p.add_argument('--owned-background-pid',type=int,action='append',default=[],
                   help='Explicit previously audited own display/remote-control PID; not an exclusive performance run')
    args=p.parse_args()
    onnx,plugin,target=Path(args.onnx),Path(args.plugin),Path(args.output)
    if target.exists():p.error('refusing to overwrite an engine')
    if sha256(onnx)!=args.onnx_sha256 or sha256(plugin)!=args.plugin_sha256:p.error('ONNX/plugin SHA mismatch')
    import torch
    import tensorrt as trt
    if not torch.cuda.is_available():p.error('available CUDA Torch runtime required')
    hardware=host_resources(args.device)
    index=(hardware.get('gpu') or {}).get('index')
    if index is None:p.error('GPU ownership cannot be established')
    active=subprocess.run(['nvidia-smi','-i',index,'--query-compute-apps=pid','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,timeout=10,check=True)
    pids={int(x.strip()) for x in active.stdout.splitlines() if x.strip().isdigit()}
    if pids-set(args.owned_background_pid):p.error('GPU has an unaudited compute workload; do not interrupt or share it for building')
    with torch.cuda.device(args.device):
        ctypes.CDLL(str(plugin.resolve()),mode=getattr(ctypes,'RTLD_GLOBAL',0))
        log=trt.Logger(trt.Logger.WARNING);trt.init_libnvinfer_plugins(log,'')
        builder=trt.Builder(log);network=builder.create_network(0);parser=trt.OnnxParser(network,log)
        if not parser.parse(onnx.read_bytes()):
            raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
        shapes={network.get_input(i).name:list(network.get_input(i).shape) for i in range(network.num_inputs)}
        if shapes!={'input_left':[1,3,288,480],'input_right':[1,3,288,480]}:
            raise RuntimeError('unvalidated LightESS input ABI: '+str(shapes))
        cfg=builder.create_builder_config();cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,int(args.workspace_gib*2**30))
        cfg.set_flag(trt.BuilderFlag.FP16);cfg.builder_optimization_level=3
        started=time.perf_counter();plan=builder.build_serialized_network(network,cfg)
        if plan is None:raise RuntimeError('TensorRT build failed')
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(bytes(plan))
        contract=dict(path=str(target.resolve()),sha256=sha256(target),kind='trt_engine',source='local build from pinned public ONNX',
                      compatibility=dict(compute_capability=hardware['gpu']['compute_capability'],tensorrt_version=trt.__version__,
                                         precision='FP16 model / FP32 geometry',model_sha256=args.onnx_sha256,shape=shapes,
                                         plugin_sha256=args.plugin_sha256,cuda_version=torch.version.cuda))
        atomic_json(target.with_suffix(target.suffix+'.json'),dict(asset=contract,hardware=hardware,build_wall_s=time.perf_counter()-started,
                    active_gpu_pids_before_build=sorted(pids),declared_owned_background_pids=args.owned_background_pid,
                    # Graphics/remote-control work can be absent from compute-apps.
                    # A declared background owner still makes this nonexclusive.
                    exclusive_performance_run=not pids and not args.owned_background_pid,
                    execution_parity_checked=False,full_pipeline_accepted=False))
        print(json.dumps(contract))


if __name__=='__main__':main()
