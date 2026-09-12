#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch import nn
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.model_loader.reload import (
    finalize_layerwise_reload,
    initialize_layerwise_reload,
    record_metadata_for_reloading,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_ascend.ops.kimi_kda import (
    _PACKED_CONV_WEIGHT_NAME,
    AscendKimiGatedDeltaNetAttention,
    _additional_config_flag,
    _graph_safe_cache_indices,
    _graph_safe_cache_update,
    _load_a_log,
    _zero_padded_spec_output,
)


class _NoopQuantMethod(QuantizeMethodBase):
    def create_weights(self, layer: nn.Module, *args, **kwargs):
        raise NotImplementedError

    def apply(self, layer: nn.Module, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


def test_kimi_options_are_read_from_vllm_additional_config():
    config = SimpleNamespace(additional_config={"kimi_training_causal_conv1d": True})

    assert _additional_config_flag(config, "kimi_training_causal_conv1d")
    assert not _additional_config_flag(config, "missing")
    assert _additional_config_flag(SimpleNamespace(additional_config={"flag": 1}), "flag")
    with pytest.raises(ValueError, match="must be a boolean"):
        _additional_config_flag(SimpleNamespace(additional_config={"flag": "yes"}), "flag")
    with pytest.raises(ValueError, match="must be a boolean"):
        _additional_config_flag(SimpleNamespace(additional_config={"flag": 1.0}), "flag")


def test_kimi_kda_format_cast_is_fake_tensor_safe():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        value = torch.empty(2, 3, 4, dtype=torch.bfloat16).transpose(0, 1)
        result = torch.ops.vllm.kimi_kda_npu_format_cast_nd(value)

    assert result.shape == value.shape
    assert result.stride() == value.stride()
    assert result.dtype == value.dtype
    assert result.device == value.device


def _make_conv_pack_attention(
    *,
    local_num_heads: int = 2,
    head_dim: int = 3,
    conv_size: int = 4,
    model_dtype: torch.dtype = torch.bfloat16,
) -> AscendKimiGatedDeltaNetAttention:
    attention = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    attention.local_num_heads = local_num_heads
    attention.head_dim = head_dim
    attention.conv_size = conv_size
    attention.model_config = SimpleNamespace(dtype=model_dtype)

    local_channels = local_num_heads * head_dim
    for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
        conv = nn.Module()
        conv.weight = nn.Parameter(
            torch.empty(
                local_channels,
                1,
                conv_size,
                dtype=torch.float32,
            )
        )
        conv.weight.weight_loader = default_weight_loader
        conv.quant_method = _NoopQuantMethod()
        setattr(attention, name, conv)

    attention.q_conv1d.register_parameter(
        _PACKED_CONV_WEIGHT_NAME,
        nn.Parameter(
            torch.empty(
                attention._packed_conv_shape(),
                dtype=model_dtype,
            ),
            requires_grad=False,
        ),
    )
    for conv in (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d):
        attention._wrap_conv_process_weights(conv)
    return attention


def _process_conv_weights(
    attention: AscendKimiGatedDeltaNetAttention,
) -> None:
    for conv in (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d):
        conv.quant_method.process_weights_after_loading(conv)


def _expected_packed_conv_weights(
    attention: AscendKimiGatedDeltaNetAttention,
) -> torch.Tensor:
    return torch.cat(
        [
            conv.weight[:, 0, :].transpose(0, 1)
            for conv in (
                attention.q_conv1d,
                attention.k_conv1d,
                attention.v_conv1d,
            )
        ],
        dim=1,
    ).to(attention.model_config.dtype)


def test_load_a_log_slices_padded_1d_checkpoint_by_tp_rank():
    param = torch.empty(1, 1, 2, 1)
    loaded_weight = torch.arange(6, dtype=torch.float32)

    with patch("vllm_ascend.ops.kimi_kda.get_tensor_model_parallel_rank", return_value=1):
        _load_a_log(param, loaded_weight, num_heads=4)

    torch.testing.assert_close(param, torch.tensor([[[[2.0], [3.0]]]]))


def test_load_a_log_preserves_exact_local_4d_checkpoint():
    param = torch.empty(1, 1, 2, 1)
    loaded_weight = torch.tensor([[[[4.0], [5.0]]]])

    _load_a_log(param, loaded_weight, num_heads=4)

    torch.testing.assert_close(param, loaded_weight)


def test_load_a_log_rejects_unsupported_layout():
    with pytest.raises(ValueError, match="must be 1-D or 4-D"):
        _load_a_log(torch.empty(1, 1, 2, 1), torch.empty(2, 2), num_heads=4)


def test_zero_padded_spec_output_clears_uninitialized_tail():
    output = torch.arange(16 * 2 * 3, dtype=torch.float32).reshape(1, 16, 2, 3)
    output[:, 8:] = torch.nan
    query_start_loc = torch.tensor([0, 8, 8], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked[:, :8], output[:, :8])
    assert torch.equal(masked[:, 8:], torch.zeros_like(masked[:, 8:]))
    assert torch.isfinite(masked).all()


def test_zero_padded_spec_output_preserves_fully_covered_output():
    output = torch.randn(1, 16, 2, 3)
    query_start_loc = torch.tensor([0, 8, 16], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked, output)


def test_zero_padded_spec_output_supports_multiple_real_and_dummy_rows():
    output = torch.randn(1, 32, 2, 3)
    expected = output[:, :16].clone()
    output[:, 16:] = torch.nan
    query_start_loc = torch.tensor([0, 8, 16, 16, 16], dtype=torch.int32)

    masked = _zero_padded_spec_output(output, query_start_loc)

    torch.testing.assert_close(masked[:, :16], expected)
    assert torch.equal(masked[:, 16:], torch.zeros_like(masked[:, 16:]))
    assert masked.shape == output.shape
    assert masked.dtype == output.dtype
    assert masked.device == output.device


def test_graph_safe_cache_update_uses_static_indices_and_preserves_padding():
    cache_indices = torch.tensor([3, 5, -1, -1])
    valid, safe_indices = _graph_safe_cache_indices(cache_indices)
    assert torch.equal(valid, torch.tensor([True, True, False, False]))
    assert torch.equal(safe_indices, torch.tensor([3, 5, 3, 3]))

    cache = torch.zeros(8, 2, 3)
    values = torch.arange(4 * 2 * 3, dtype=cache.dtype).reshape(4, 2, 3)
    _graph_safe_cache_update(cache, safe_indices, valid, values)

    torch.testing.assert_close(cache[3], values[0])
    torch.testing.assert_close(cache[5], values[1])
    assert torch.equal(cache[[0, 1, 2, 4, 6, 7]], torch.zeros(6, 2, 3))


def test_output_norm_gate_uses_kda_fused_triton_kernel():
    attention = AscendKimiGatedDeltaNetAttention.__new__(AscendKimiGatedDeltaNetAttention)
    nn.Module.__init__(attention)
    attention.o_norm = SimpleNamespace(
        weight=nn.Parameter(torch.randn(3)),
        eps=1e-6,
    )
    core_attn_out = torch.randn(1, 4, 2, 3)
    output_gate = torch.randn(4, 2, 3)
    expected = torch.randn_like(core_attn_out)

    with patch(
        "vllm_ascend.ops.kimi_kda.apply_kda_rms_norm_sigmoid_gate",
        return_value=expected,
    ) as fused_norm_gate:
        actual = attention._apply_output_norm_gate(core_attn_out, output_gate)

    assert actual is expected
    fused_norm_gate.assert_called_once_with(
        core_attn_out,
        output_gate,
        attention.o_norm.weight,
        attention.o_norm.eps,
    )


def test_conv_post_load_processing_packs_kernel_layout_in_place():
    attention = _make_conv_pack_attention()
    convs = (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    for shard_id, conv in enumerate(convs):
        conv.weight.data.copy_(
            torch.arange(
                conv.weight.numel(),
                dtype=torch.float32,
            ).reshape_as(conv.weight)
            + shard_id * 100
        )

    packed = attention.q_conv1d.get_parameter(_PACKED_CONV_WEIGHT_NAME)
    original_ptr = packed.data_ptr()
    _process_conv_weights(attention)

    assert packed.data_ptr() == original_ptr
    torch.testing.assert_close(packed, _expected_packed_conv_weights(attention))
    assert packed.dtype == torch.bfloat16
    assert packed.is_contiguous()
    assert attention._conv_weights_t().data_ptr() == original_ptr
    parameter_name = f"q_conv1d.{_PACKED_CONV_WEIGHT_NAME}"
    assert dict(attention.named_parameters())[parameter_name] is packed
    assert attention.state_dict()[parameter_name].data_ptr() == original_ptr

    convs[1].weight.data.fill_(777)
    convs[1].quant_method.process_weights_after_loading(convs[1])

    assert attention._conv_weights_t().data_ptr() == original_ptr
    torch.testing.assert_close(
        attention._conv_weights_t(),
        _expected_packed_conv_weights(attention),
    )


def test_full_checkpoint_reload_refreshes_packed_weight_in_place():
    attention = _make_conv_pack_attention()
    convs = (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    for shard_id, conv in enumerate(convs, start=1):
        conv.weight.data.fill_(shard_id)
    _process_conv_weights(attention)
    original_packed = attention._conv_weights_t()
    original_ptr = original_packed.data_ptr()

    record_metadata_for_reloading(attention)
    initialize_layerwise_reload(attention)
    for shard_id, conv in enumerate(convs, start=11):
        loaded_weight = torch.full(
            conv.weight.shape,
            shard_id,
            dtype=torch.float32,
        )
        conv.weight.weight_loader(conv.weight, loaded_weight)
    finalize_layerwise_reload(
        attention,
        SimpleNamespace(dtype=torch.bfloat16),
    )

    refreshed = attention._conv_weights_t()
    assert refreshed is original_packed
    assert refreshed.data_ptr() == original_ptr
    torch.testing.assert_close(
        refreshed,
        _expected_packed_conv_weights(attention),
    )


def test_repack_waits_until_all_source_weights_are_materialized():
    attention = _make_conv_pack_attention()
    source_convs = (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    for value, conv in enumerate(source_convs, start=1):
        conv.weight.data.fill_(value)
    packed = attention._conv_weights_t()
    packed.data.fill_(-1)
    before = packed.clone()
    v_shape = attention.v_conv1d.weight.shape
    attention.v_conv1d.weight = nn.Parameter(
        torch.empty(v_shape, device="meta"),
    )

    attention.q_conv1d.quant_method.process_weights_after_loading(attention.q_conv1d)

    torch.testing.assert_close(packed, before)

    attention.v_conv1d.weight = nn.Parameter(
        torch.full(v_shape, 3, dtype=torch.float32),
    )
    attention.v_conv1d.quant_method.process_weights_after_loading(attention.v_conv1d)

    torch.testing.assert_close(packed, _expected_packed_conv_weights(attention))


def test_kernel_format_reload_updates_named_packed_parameter():
    attention = _make_conv_pack_attention(model_dtype=torch.float16)
    for shard_id, conv in enumerate(
        (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d),
        start=1,
    ):
        conv.weight.data.fill_(shard_id)
    _process_conv_weights(attention)

    parameter_name = f"q_conv1d.{_PACKED_CONV_WEIGHT_NAME}"
    packed = attention.get_parameter(parameter_name)
    original_ptr = packed.data_ptr()
    kernel_weight = torch.arange(
        packed.numel(),
        dtype=packed.dtype,
    ).reshape_as(packed)

    with torch.no_grad():
        attention.get_parameter(parameter_name).copy_(kernel_weight)

    assert attention._conv_weights_t().data_ptr() == original_ptr
    torch.testing.assert_close(attention._conv_weights_t(), kernel_weight)


@pytest.mark.parametrize("backend", ["fsdpturbo", "megatron"])
@pytest.mark.parametrize("run_mode", [0, 1])
def test_training_convolution_dispatch_preserves_cache_and_sequence_layout(backend, run_mode):
    attention = _make_conv_pack_attention()
    attention.use_training_causal_conv1d = True
    attention.training_backend = backend
    mixed = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    weights = torch.ones(4, 18)
    cache = torch.ones(3, 3, 18)
    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, 1, 2]), cache_indices=torch.tensor([2, 0]))

    def conv(*, x, weight, initial_state, activation, cu_seqlens, output_final_state):
        assert x.shape == ((2, 1, 6) if run_mode else (1, 2, 6))
        assert (cu_seqlens is None) == bool(run_mode)
        assert activation == "silu" and output_final_state
        torch.testing.assert_close(initial_state[..., 0], torch.zeros(2, 6))
        torch.testing.assert_close(initial_state[..., 1:], torch.ones(2, 6, 3))
        return x + 1, initial_state + 2

    selected = Mock(side_effect=conv)
    other = Mock(side_effect=AssertionError("wrong training backend"))
    modules = {
        "fsdp_turbo.ops.causal_conv1d": SimpleNamespace(causal_conv1d=selected if backend == "fsdpturbo" else other),
        "mindspeed_ops.api.triton.convolution": SimpleNamespace(
            causal_conv1d=selected if backend == "megatron" else other
        ),
    }
    with patch.dict(sys.modules, modules):
        output = attention._run_causal_conv1d(mixed, weights, cache, metadata, run_mode=run_mode)
    torch.testing.assert_close(output, mixed + 1)
    torch.testing.assert_close(cache[[2, 0]], torch.full((2, 3, 18), 3.0))
    torch.testing.assert_close(cache[1], torch.ones(3, 18))
    assert selected.call_count == 3
    other.assert_not_called()
