#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Protect deferred layerwise weights from Verl's reusable IPC buffer."""

from collections.abc import Callable
from functools import wraps

import torch
from vllm.model_executor.model_loader.reload import layerwise as layerwise_reload

_make_online_process_loader = layerwise_reload.make_online_process_loader


def _wrap_parameters_weight_loader(layer: torch.nn.Module) -> None:
    """Wrap partial and other callable loaders without assuming __name__."""
    for name, tensor in layerwise_reload.get_layer_tensors(layer).items():
        if name in layerwise_reload.SKIP_TENSORS:
            continue
        loader = layerwise_reload._get_weight_loader(tensor)
        if getattr(loader, "__name__", None) != "online_process_loader":
            tensor.weight_loader = layerwise_reload.make_online_process_loader(layer, name)


def _get_original_loader(tensor: torch.Tensor) -> Callable:
    """Remove layerwise wrappers from an arbitrary callable loader."""
    loader = layerwise_reload._get_weight_loader(tensor)
    while getattr(loader, "__name__", None) == "online_process_loader":
        loader = loader.__wrapped__
    return loader


def _make_online_process_loader_with_owned_tensors(
    layer: torch.nn.Module,
    param_name: str,
) -> Callable:
    """Own source tensors that vLLM defers beyond one reload call.

    Verl receives each weight bucket as views into one reusable IPC buffer.
    vLLM's layerwise loader may retain a shard until later shards arrive in a
    subsequent bucket, so keeping the view would let the sender overwrite it.
    Clone only calls that remain deferred; completed parameters are processed
    and released by the upstream loader before this wrapper returns.
    """
    loader = _make_online_process_loader(layer, param_name)

    @wraps(loader)
    def online_process_loader(*args, **kwargs):
        info = layerwise_reload.get_layerwise_info(layer)
        loaded_before = info.load_numel if info.can_load() else None
        result = loader(*args, **kwargs)
        info = layerwise_reload.get_layerwise_info(layer)
        if not info.can_load() or not info.loaded_weights:
            return result
        if loaded_before is None or info.load_numel <= loaded_before:
            return result

        loaded_name, bound_args = info.loaded_weights[-1]
        if loaded_name != param_name:
            return result
        for name, value in bound_args.arguments.items():
            if name != "param" and isinstance(value, torch.Tensor) and value.device.type != "meta":
                bound_args.arguments[name] = value.detach().clone()
        return result

    return online_process_loader


layerwise_reload._wrap_parameters_weight_loader = _wrap_parameters_weight_loader
layerwise_reload._get_original_loader = _get_original_loader
layerwise_reload.make_online_process_loader = _make_online_process_loader_with_owned_tensors
