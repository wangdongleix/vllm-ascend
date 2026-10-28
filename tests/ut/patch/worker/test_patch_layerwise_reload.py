# SPDX-License-Identifier: Apache-2.0
from functools import partial

import pytest
import torch
from torch import nn
from vllm.model_executor.model_loader.reload import layerwise

import vllm_ascend.patch.worker.patch_layerwise_reload  # noqa: F401


def test_partial_loaders_reuse_caller_owned_weights_and_keep_kernel_storage():
    layer = nn.Linear(3, 2)

    def load(param, loaded_weight):
        param.data.copy_(loaded_weight)

    layer.weight.weight_loader = partial(load)
    layer.bias.weight_loader = partial(load)
    layerwise.record_metadata_for_reloading(layer)
    pointer = layer.weight.data_ptr()
    buffer = torch.empty(6)
    for version in (0, 1):
        expected_weight = torch.arange(6, dtype=torch.float32).reshape(2, 3) + version * 10
        expected_bias = torch.tensor([3.0, 5.0]) + version
        layerwise.initialize_layerwise_reload(layer)
        buffer.copy_(expected_weight.flatten())
        # Verl owns the IPC view once, before calling the upstream loader.
        owned_weight = buffer.view(2, 3).clone()
        layer.weight.weight_loader(layer.weight, owned_weight)
        _, bound_args = layerwise.get_layerwise_info(layer).loaded_weights[-1]
        assert bound_args.arguments["loaded_weight"].data_ptr() == owned_weight.data_ptr()
        buffer.fill_(-999)
        layer.bias.weight_loader(layer.bias, expected_bias)
        torch.testing.assert_close(layer.weight, expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(layer.bias, expected_bias, rtol=0, atol=0)
        assert layer.weight.data_ptr() == pointer


@pytest.mark.parametrize("buffer_name", ["log2phy", "moe_load", "load_counter", "expert_ids_per_ep_rank"])
@pytest.mark.parametrize("persistent", [True, False])
def test_runtime_routing_buffers_do_not_delay_reload_or_lose_live_state(buffer_name, persistent):
    layer = nn.Linear(3, 2, bias=False)
    runtime_state = torch.arange(16, dtype=torch.int64)
    layer.register_buffer(buffer_name, runtime_state, persistent=persistent)
    layerwise.record_metadata_for_reloading(layer)
    weight_pointer = layer.weight.data_ptr()
    for version in (1, 2):
        # Routing state may change after the initial metadata snapshot.
        runtime_state.add_(version)
        expected_state = runtime_state.clone()
        layerwise.initialize_layerwise_reload(layer)
        assert getattr(layer, buffer_name) is runtime_state
        expected_weight = torch.full((2, 3), float(version))
        layer.weight.weight_loader(layer.weight, expected_weight)
        info = layerwise.get_layerwise_info(layer)
        assert not info.can_load()
        assert not info.loaded_weights
        assert layer.weight.data_ptr() == weight_pointer
        assert getattr(layer, buffer_name) is runtime_state
        assert (buffer_name not in layer._non_persistent_buffers_set) == persistent
        torch.testing.assert_close(layer.weight, expected_weight, rtol=0, atol=0)
        torch.testing.assert_close(runtime_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("parent_first", [True, False])
def test_kda_derived_conv_reload_preserves_new_values_and_kernel_storage(parent_first):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from vllm_ascend.ops.kimi_kda import _PACKED_CONV_WEIGHT_NAME, AscendKimiK3DeltaAttention

    attention = AscendKimiK3DeltaAttention.__new__(AscendKimiK3DeltaAttention)
    nn.Module.__init__(attention)
    attention.conv1d = nn.Module()
    attention.conv1d.weight = nn.Parameter(torch.zeros(6, 1, 4), requires_grad=False)
    attention.register_buffer(_PACKED_CONV_WEIGHT_NAME, torch.zeros(4, 6), persistent=False)
    attention.A_log = nn.Parameter(torch.zeros(2), requires_grad=False)
    attention.conv1d.quant_method = UnquantizedLinearMethod()
    attention.conv1d.quant_method.process_weights_after_loading = lambda layer: attention._pack_conv_weights()
    layerwise.record_metadata_for_reloading(attention)
    pointer = attention.get_buffer(_PACKED_CONV_WEIGHT_NAME).data_ptr()
    for version in (1, 2):
        layerwise.initialize_layerwise_reload(attention)
        source = torch.arange(24, dtype=torch.float32).reshape(6, 1, 4) + version
        expected = source[:, 0, :].T.clone()
        if parent_first:
            attention.A_log.weight_loader(attention.A_log, torch.full((2,), float(version)))
        attention.conv1d.weight.weight_loader(attention.conv1d.weight, source)
        source.fill_(-999)
        if not parent_first:
            attention.A_log.weight_loader(attention.A_log, torch.full((2,), float(version)))
        layerwise.finalize_layerwise_reload(attention, None)
        packed = attention.get_buffer(_PACKED_CONV_WEIGHT_NAME)
        torch.testing.assert_close(packed, expected, rtol=0, atol=0)
        assert packed.data_ptr() == pointer
