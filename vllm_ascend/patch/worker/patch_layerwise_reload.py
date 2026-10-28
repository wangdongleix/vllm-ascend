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

"""Adapt upstream layerwise reload for Ascend loaders and runtime tensors."""

from collections.abc import Callable

import torch
from vllm.model_executor.model_loader.reload import layerwise as layerwise_reload
from vllm.model_executor.model_loader.reload import meta as reload_meta

# Ascend routing/communication state is restored by level-2 wake-up, never
# supplied by a checkpoint. Match upstream's handling of its expert maps:
# exclude these tensors from both reload accounting and meta materialization.
# KDA packed convolution weights are derived by the child convolution loader;
# preserve their live kernel storage so that parent finalization cannot restore
# stale values over the freshly packed weights.
_ASCEND_NON_CHECKPOINT_TENSORS = {
    "log2phy",
    "moe_load",
    "load_counter",
    "expert_ids_per_ep_rank",
    "ascend_conv1d_weight",
}
reload_meta.SKIP_LOAD_TENSORS.update(_ASCEND_NON_CHECKPOINT_TENSORS)
reload_meta.SKIP_TENSORS.update(_ASCEND_NON_CHECKPOINT_TENSORS)


def _wrap_parameters_weight_loader(layer: torch.nn.Module) -> None:
    """Wrap partial and other callable loaders without assuming __name__."""
    for name, tensor in layerwise_reload.get_layer_tensors(layer).items():
        if name in layerwise_reload.SKIP_LOAD_TENSORS:
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


layerwise_reload._wrap_parameters_weight_loader = _wrap_parameters_weight_loader
layerwise_reload._get_original_loader = _get_original_loader
