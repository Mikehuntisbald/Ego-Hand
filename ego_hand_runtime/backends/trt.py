"""TensorRT10 runner with exclusive contexts and explicit stream lifetimes."""
import ctypes
import numpy as np
from ..assets import require_asset
from ..errors import RuntimeFailure


class Engine:
    def __init__(self, config, asset, plugin=None):
        import torch
        import tensorrt as trt
        self.torch, self.trt, self.device = torch, trt, config['device']
        if not torch.cuda.is_available():
            raise RuntimeFailure('CUDA', 'TensorRT requires an available CUDA device')
        contract=config['assets'][asset].get('compatibility',{})
        if plugin and contract.get('plugin_sha256')!=config['assets'][plugin].get('sha256'):
            raise RuntimeFailure('engine_plugin','Engine/plugin SHA binding differs')
        if contract.get('cuda_version')!=torch.version.cuda:
            raise RuntimeFailure('engine_CUDA','Engine was built for a different CUDA runtime contract')
        if plugin:
            self.plugin = ctypes.CDLL(str(require_asset(config, plugin)), mode=getattr(ctypes, 'RTLD_GLOBAL', 0))
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        with torch.cuda.device(self.device):
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(require_asset(config, asset).read_bytes())
            if self.engine is None:
                raise RuntimeFailure('engine', 'TensorRT engine could not be deserialized')
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.Stream(device=self.device)
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.buffers = {}

    def __call__(self, inputs):
        torch, trt = self.torch, self.trt
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            for name in self.names:
                if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:
                    if name not in inputs:
                        raise RuntimeFailure('engine_input', 'Missing declared engine input', name=name)
                    x = np.asarray(inputs[name])
                    if not self.context.set_input_shape(name, x.shape):
                        raise RuntimeFailure('engine_shape', 'Input is outside the engine shape profile')
            for name in self.names:
                shape = tuple(self.context.get_tensor_shape(name))
                if any(x<0 for x in shape):
                    raise RuntimeFailure('engine_shape', 'Unresolved TensorRT output dimensions')
                dtype = torch.from_numpy(np.empty((), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))).dtype
                if name not in self.buffers or self.buffers[name].shape!=shape:
                    self.buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
                self.context.set_tensor_address(name, self.buffers[name].data_ptr())
                if name in inputs:
                    self.buffers[name].copy_(torch.as_tensor(np.ascontiguousarray(inputs[name]), device=self.device))
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeFailure('engine_execute', 'TensorRT execution failed')
        # The buffers are owned by the exclusive model slot. Synchronize before
        # CPU copies and before returning the slot to another request.
        self.stream.synchronize()
        return {n: self.buffers[n].float().cpu().numpy().copy() for n in self.names
                if self.engine.get_tensor_mode(n)==trt.TensorIOMode.OUTPUT}
