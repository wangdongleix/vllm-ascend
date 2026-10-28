# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from vllm.v1.worker.utils import clear_layer_kv_caches

from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import AscendKimiMLAAttention, KimiTrainingPatchEmbed, _load_packed_expert_weight


@pytest.mark.parametrize("with_standard_weight", [False, True])
def test_packed_buckets_skip_empty_upstream_load_and_preserve_mixed_stream(with_standard_weight):
    model = kimi_k3.AscendKimiLinearModel.__new__(kimi_k3.AscendKimiLinearModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=2, is_moe=True)
    model.n_redundant_experts = 0
    model.named_parameters = MagicMock(side_effect=AssertionError("unnecessary full-model traversal"))
    packed = ("layers.0.block_sparse_moe.experts.gate_up_proj.__verl_packed_local__.0", torch.ones(2, 4, 6))
    standard = ("layers.0.self_attn.q_proj.weight", torch.ones(4, 4), {"loaded_shard_id": 0})
    weights = [packed, standard, packed] if with_standard_weight else [packed]
    received = []

    def upstream_load(self, stream):
        received.extend(stream)
        return {row[0] for row in received}

    with (
        patch.object(kimi_k3, "_load_packed_expert_weight", return_value="packed_target") as packed_load,
        patch.object(
            kimi_k3.UpstreamKimiLinearModel, "load_weights", autospec=True, side_effect=upstream_load
        ) as upstream,
    ):
        loaded = model.load_weights(iter(weights))

    assert packed_load.call_count == (2 if with_standard_weight else 1)
    assert upstream.call_count == int(with_standard_weight)
    assert packed_load.call_args.args[2] == model.get_parameter
    assert received == ([standard] if with_standard_weight else [])
    assert loaded == ({"packed_target", standard[0]} if with_standard_weight else {"packed_target"})
    model.named_parameters.assert_not_called()


@pytest.mark.parametrize("tensor_cache", [True, False])
def test_kimi_mla_shutdown_clears_underlying_cache(tensor_cache):
    layer = AscendKimiMLAAttention.__new__(AscendKimiMLAAttention)
    nn.Module.__init__(layer)
    cache = torch.ones(4) if tensor_cache else [torch.ones(4)]
    inner = SimpleNamespace(kv_cache=cache, impl=SimpleNamespace(_k_scale_cache=cache, _v_scale_cache=cache))
    layer.mla_attn = SimpleNamespace(mla_attn=inner)

    clear_layer_kv_caches([layer])

    assert layer.kv_cache is inner.kv_cache
    assert len(inner.kv_cache) == 0
    assert isinstance(inner.kv_cache, torch.Tensor) == tensor_cache
    assert inner.impl._k_scale_cache is None
    assert inner.impl._v_scale_cache is None


@pytest.mark.parametrize("gate_up", [True, False])
def test_packed_local_reload_preserves_expert_ids_axes_and_storage(gate_up):
    source = torch.arange(2 * 7 * 6, dtype=torch.float32).reshape(2, 7, 6)
    target = nn.Parameter(torch.empty(2, 6, 7), requires_grad=False)
    calls = []

    def load(param, shard, name, *, expert_id, shard_id, return_success):
        calls.append((expert_id, shard_id))
        index = expert_id - 256
        begin = 0 if shard_id in ("w1", "w2") else 3
        param[index, begin : begin + shard.shape[0]].copy_(shard)
        return True

    target.weight_loader = load
    prefix = "layers.1.block_sparse_moe.experts."
    source_name = prefix + ("gate_up_proj" if gate_up else "down_proj")
    target_name = prefix + "routed_experts." + ("w13_weight" if gate_up else "w2_weight")
    pointer = target.data_ptr()
    for version in (0, 1):
        weight = source + version * 1000
        assert (
            _load_packed_expert_weight(
                source_name + ".__verl_packed_local__.256",
                weight,
                {target_name: target}.__getitem__,
                896,
            )
            == target_name
        )
        torch.testing.assert_close(target, weight.transpose(1, 2), rtol=0, atol=0)
        assert target.data_ptr() == pointer
    assert {expert for expert, _ in calls} == {256, 257}


def test_patch_projection_matches_actor_fp32_linear_and_retains_names():
    projection = nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False).to(torch.bfloat16)
    original = nn.Module()
    original.proj = projection
    original.input_norm = lambda value, dtype: value.to(dtype)
    original.pos_emb = lambda value, grid: value
    patch = KimiTrainingPatchEmbed(original)
    x = torch.randn(5, 3, 2, 2).to(torch.bfloat16)
    expected = F.linear(x.flatten(1).float(), projection.weight.flatten(1).float()).to(torch.bfloat16)
    torch.testing.assert_close(patch(x, [[1, 1, 5]]), expected, rtol=0, atol=0)
    assert dict(patch.named_parameters())["proj.weight"] is projection.weight


def test_ascend_kimi_requires_processor_side_image_normalization():
    from vllm_ascend.models.kimi_k3 import AscendKimiK3ForConditionalGeneration

    # A uint8 processor path requires a device normalization layer. The Ascend
    # tower instead consumes the same normalized patches as the training model.
    assert AscendKimiK3ForConditionalGeneration.supports_mm_device_do_normalize is False
