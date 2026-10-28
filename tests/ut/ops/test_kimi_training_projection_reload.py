# SPDX-License-Identifier: Apache-2.0
from unittest.mock import patch

import pytest
import torch
from torch import nn
from vllm.model_executor.model_loader.reload import layerwise

import vllm_ascend.patch.worker.patch_layerwise_reload  # noqa: F401
from vllm_ascend.ops.kimi_kda import _KimiTrainingInputProjection, _KimiTrainingOutputProjection


@pytest.mark.parametrize("reduce_results", [False, True])
def test_output_projection_reduces_once_with_sequence_parallel(reduce_results):
    layer = _KimiTrainingOutputProjection.__new__(_KimiTrainingOutputProjection)
    nn.Module.__init__(layer)
    layer.tp_size = 8
    layer.reduce_results = reduce_results
    layer.weight = nn.Parameter(torch.ones(4, 2, dtype=torch.bfloat16), requires_grad=False)
    hidden_states = torch.ones(8, 2, dtype=torch.bfloat16)
    with patch(
        "vllm_ascend.ops.kimi_kda.tensor_model_parallel_all_reduce", side_effect=lambda value: value * 8
    ) as all_reduce:
        output, bias = layer(hidden_states)
    assert all_reduce.call_count == int(reduce_results)
    assert bias is None
    # Eight identical rank contributions: the decoder performs reduce-scatter
    # only when the projection leaves its local partial result unreduced.
    reduced = output if reduce_results else (output * 8).chunk(8)[0]
    torch.testing.assert_close(reduced, torch.full_like(reduced, 16), rtol=0, atol=0)


def test_padded_tp_projection_finishes_each_reload_without_model_flush():
    tp_size, tp_rank, hidden_size, padding = 8, 3, 8, 15
    checkpoint_sizes = [16, 16, 16, 16, 8, 8]
    with (
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_world_size", return_value=tp_size),
        patch("vllm.model_executor.layers.linear.get_tensor_model_parallel_rank", return_value=tp_rank),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_rank", return_value=tp_rank),
        patch("vllm.model_executor.parameter.get_tensor_model_parallel_world_size", return_value=tp_size),
    ):
        layer = _KimiTrainingInputProjection(
            hidden_size,
            checkpoint_sizes + [padding * tp_size],
            replicated_shard_id=4,
            tp_size=tp_size,
            padding_size=padding,
            bias=False,
            params_dtype=torch.float32,
        )
    layerwise.record_metadata_for_reloading(layer)
    pointer = layer.weight.data_ptr()
    for version in (0, 1):
        layerwise.initialize_layerwise_reload(layer)
        expected = []
        for shard_id, rows in enumerate(checkpoint_sizes):
            source = torch.arange(rows * hidden_size, dtype=torch.float32).view(rows, hidden_size)
            source += version * 1000 + shard_id * 100
            local_rows = rows if shard_id == 4 else rows // tp_size
            offset = 0 if shard_id == 4 else tp_rank * local_rows
            expected.append(source[offset : offset + local_rows].clone())
            layer.weight.weight_loader(layer.weight, source, shard_id)
            source.fill_(-999)  # The IPC bucket may be reused immediately.
        info = layerwise.get_layerwise_info(layer)
        assert not info.loaded_weights
        assert not info.can_load()
        assert layer.weight.data_ptr() == pointer
        expected.append(torch.zeros(padding, hidden_size))
        torch.testing.assert_close(layer.weight, torch.cat(expected), rtol=0, atol=0)
