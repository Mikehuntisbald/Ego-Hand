"""Build the public LightESS ONNX for H20; no foreign GPU plan is loaded."""
import ctypes
import hashlib
import json
from pathlib import Path
import time
import tensorrt as trt

ROOT = Path('/mnt/why/public_headhand_h20_20261019')


def main():
    source = ROOT/'assets/ess_legacy_public/dnn_stereo_disparity_v4.1.0_onnx'
    target = ROOT/'assets/lightess_h20_fp16.engine'
    assert not target.exists(), 'Do not overwrite an existing engine'
    plugin = source/'plugins/x86_64/ess_plugins.so'
    ctypes.CDLL(str(plugin), mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    assert parser.parse((source/'light_ess.onnx').read_bytes()), [str(parser.get_error(i)) for i in range(parser.num_errors)]
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.builder_optimization_level = 3
    started = time.time()
    engine = builder.build_serialized_network(network, config)
    assert engine is not None
    with target.open('xb') as stream:
        stream.write(bytes(engine))
    result = dict(complete=True, TRT=trt.__version__, shape=[1, 3, 288, 480],
                  onnx_sha256=hashlib.file_digest((source/'light_ess.onnx').open('rb'), 'sha256').hexdigest(),
                  plugin_sha256=hashlib.file_digest(plugin.open('rb'), 'sha256').hexdigest(),
                  engine_sha256=hashlib.file_digest(target.open('rb'), 'sha256').hexdigest(),
                  seconds=time.time()-started, target='H20 SM90, not the 5090 production engine')
    (ROOT/'receipts/lightess_build.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
