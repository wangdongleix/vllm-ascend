# SPDX-License-Identifier: Apache-2.0
"""Refresh private NZ parameters without invalidating captured graph addresses."""
import acl
import torch
import torch_npu
from vllm.model_executor.utils import replace_parameter


def replace_nz_parameter(layer, name, new_tensor):
    """Register once, then copy identical NZ layouts using ACL on Ascend."""
    if new_tensor.device.type != 'npu' or torch_npu.get_npu_format(new_tensor) != 29:
        raise ValueError('NZ reload requires an NPU FRACTAL_NZ tensor')
    if new_tensor.storage_offset() != 0:
        raise ValueError('NZ reload requires zero storage offset')
    old = getattr(layer, name, None)
    if old is None:
        replace_parameter(layer, name, new_tensor)
        return
    if (old.shape != new_tensor.shape or old.dtype != new_tensor.dtype
            or old.device != new_tensor.device or old.stride() != new_tensor.stride()
            or old.storage_offset() != 0 or torch_npu.get_npu_format(old) != 29
            or old.untyped_storage().nbytes() != new_tensor.untyped_storage().nbytes()):
        raise ValueError(f'Incompatible NZ reload for {name}; refusing to invalidate captured storage')
    # ACL copies physical bytes, including NZ padding. Synchronize producers
    # and graph consumers before the synchronous device-to-device transfer.
    with torch.npu.device(old.device):
        torch.npu.synchronize()
        size = new_tensor.untyped_storage().nbytes()
        status = acl.rt.memcpy(old.data_ptr(), size, new_tensor.data_ptr(), size, 3)
        if status != 0:
            raise RuntimeError(f'ACL NZ reload failed for {name}: {status}')
