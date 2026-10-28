# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ops.fused_moe.router.grouped_topk_router import AscendGroupedTopKRouter
from vllm_ascend.ops.fused_moe.router.kimi_training_router import KimiTrainingRouter
from vllm_ascend.patch.kimi_full_r3_schema import get_kimi_full_r3_shape
from vllm_ascend.patch.worker.patch_routed_experts_capture import _pack_kimi_full_r3


def test_capture_preserves_executed_weights_and_large_expert_ids():
    router = KimiTrainingRouter(
        AscendGroupedTopKRouter(
            top_k=2,
            global_num_experts=896,
            num_expert_group=1,
            topk_group=1,
            use_grouped_topk=True,
            scoring_func="sigmoid",
        )
    )
    capture = Mock()
    router.set_capture_fn(capture)
    logits = torch.full((3, 896), -20.0)
    logits[:, 256], logits[:, 895] = 1.0, 2.0
    weights, ids = router._select_experts(torch.zeros(3, 4), logits)
    assert weights.dtype == torch.bfloat16
    assert set(ids[0].tolist()) == {256, 895}
    capture.assert_called_once_with(ids, weights)
    payload = _pack_kimi_full_r3(ids, weights).to(torch.uint16).to(torch.int32)
    torch.testing.assert_close(payload[:, :2], ids)
    restored = torch.stack((payload[:, 2:4], payload[:, 4:]), dim=-1).to(torch.uint8)
    restored = restored.contiguous().view(torch.bfloat16).reshape_as(weights)
    torch.testing.assert_close(restored, weights, rtol=0, atol=0)


@pytest.mark.parametrize("model_type,width", [("kimi_k3", 48), ("other_moe", 16)])
def test_scheduler_and_worker_share_model_specific_wire_shape(model_type, width):
    model = SimpleNamespace(
        get_total_num_hidden_layers=lambda: 93,
        get_num_experts=lambda: 896,
        get_num_experts_per_tok=lambda: 16,
        hf_config=SimpleNamespace(model_type=model_type),
    )
    assert get_kimi_full_r3_shape(SimpleNamespace(model_config=model)) == (93, 896, width)
