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
#

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.patch.worker.patch_routed_experts_capture import capture


class TestRoutedExpertsCapturerCapture:
    """Unit tests for RoutedExpertsCapturer.capture method."""

    def _create_mock_capturer(self, dp_rank=0, tp_size=1):
        """Create a mock RoutedExpertsCapturer instance with required attributes."""
        capturer = MagicMock()
        capturer.dp_rank = dp_rank
        capturer.tp_size = tp_size
        capturer.device_buffer = torch.zeros((100, 10, 2), dtype=torch.int32)
        return capturer

    def _create_mock_forward_context(self, dp_metadata=None):
        """Create a mock forward context."""
        ctx = MagicMock()
        ctx.dp_metadata = dp_metadata
        return ctx

    def _create_mock_dp_metadata(self, num_tokens_across_dp_cpu):
        """Create mock DP metadata with token counts."""
        dp_metadata = MagicMock()
        dp_metadata.num_tokens_across_dp_cpu = torch.tensor(num_tokens_across_dp_cpu)
        return dp_metadata

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_single_dp(self, mock_get_ctx):
        """Test single DP scenario (no data parallelism)."""
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=1)
        ctx = self._create_mock_forward_context(dp_metadata=None)
        mock_get_ctx.return_value = ctx

        topk_ids = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)

        capture(capturer, layer_id=0, topk_ids=topk_ids)

        expected = topk_ids
        actual = capturer.device_buffer[:3, 0, :]
        torch.testing.assert_close(actual, expected)

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_multi_dp_naive_dispatch(self, mock_get_ctx):
        """Test multi-DP naive dispatch (n == total)."""
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=1)
        dp_metadata = self._create_mock_dp_metadata([5, 7])
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        topk_ids = torch.arange(12 * 2).view(12, 2).to(torch.int32)

        capture(capturer, layer_id=0, topk_ids=topk_ids)

        expected = topk_ids[:5]
        actual = capturer.device_buffer[:5, 0, :]
        torch.testing.assert_close(actual, expected)

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_multi_dp_modular_kernel(self, mock_get_ctx):
        """Test multi-DP modular kernel path (n == token_num_per_dp)."""
        capturer = self._create_mock_capturer(dp_rank=1, tp_size=1)
        dp_metadata = self._create_mock_dp_metadata([5, 7])
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        topk_ids = torch.arange(7 * 2).view(7, 2).to(torch.int32)

        capture(capturer, layer_id=0, topk_ids=topk_ids)

        expected = topk_ids
        actual = capturer.device_buffer[:7, 0, :]
        torch.testing.assert_close(actual, expected)

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_multi_dp_padded_all_gather(self, mock_get_ctx):
        """Test multi-DP padded all-gather path (n == total_with_padding)."""
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=1)
        dp_metadata = self._create_mock_dp_metadata([5, 7])
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        topk_ids = torch.arange(14 * 2).view(14, 2).to(torch.int32)

        capture(capturer, layer_id=0, topk_ids=topk_ids)

        expected = topk_ids[:5]
        actual = capturer.device_buffer[:5, 0, :]
        torch.testing.assert_close(actual, expected)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_tp_group")
    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.dist")
    def test_sp_modular_kernel_all2all(
        self,
        mock_dist,
        mock_get_tp_group,
        mock_get_ctx,
        mock_get_ctx1,
    ):
        """Test SP + modular kernel path with ALLTOALL comm type."""
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=2)
        dp_metadata = self._create_mock_dp_metadata([10])
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        mock_tp_group = MagicMock()
        mock_get_tp_group.return_value = mock_tp_group
        mock_tp_group.device_group = MagicMock()

        from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType

        original_comm_type = getattr(_EXTRA_CTX, "moe_comm_type", None)
        _EXTRA_CTX.moe_comm_type = MoECommType.ALLTOALL

        try:
            topk_ids = torch.arange(5 * 2).view(5, 2).to(torch.int32)

            def mock_all_gather_impl(output_list, input_tensor, device_group):
                output_list[0].copy_(input_tensor)
                output_list[1].copy_(input_tensor + 5 * 2)

            mock_dist.all_gather = mock_all_gather_impl

            capture(capturer, layer_id=0, topk_ids=topk_ids)

            expected = torch.arange(10 * 2).view(10, 2).to(torch.int32)
            actual = capturer.device_buffer[:10, 0, :]
            torch.testing.assert_close(actual, expected)
        finally:
            if original_comm_type is not None:
                _EXTRA_CTX.moe_comm_type = original_comm_type
            else:
                delattr(_EXTRA_CTX, "moe_comm_type")

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_tp_group")
    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.dist")
    def test_sp_modular_kernel_mc2(
        self,
        mock_dist,
        mock_get_tp_group,
        mock_get_ctx,
        mock_get_ctx1,
    ):
        """Test SP + modular kernel path with MC2 comm type."""
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=2)
        dp_metadata = self._create_mock_dp_metadata([5, 7])
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        mock_tp_group = MagicMock()
        mock_get_tp_group.return_value = mock_tp_group
        mock_tp_group.device_group = MagicMock()

        from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType

        original_comm_type = getattr(_EXTRA_CTX, "moe_comm_type", None)
        _EXTRA_CTX.moe_comm_type = MoECommType.MC2

        try:
            topk_ids = torch.arange(4 * 2).view(4, 2).to(torch.int32)

            def mock_all_gather_impl(output_list, input_tensor, device_group):
                output_list[0].copy_(input_tensor)
                output_list[1].copy_(input_tensor + 4 * 2)

            mock_dist.all_gather = mock_all_gather_impl

            capture(capturer, layer_id=0, topk_ids=topk_ids)

            actual = capturer.device_buffer[:5, 0, :]
            assert actual.shape[0] == 5
        finally:
            if original_comm_type is not None:
                _EXTRA_CTX.moe_comm_type = original_comm_type
            else:
                delattr(_EXTRA_CTX, "moe_comm_type")

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    def test_unexpected_batch_dim(self, mock_ascend_ctx, mock_get_ctx):
        """Test that unexpected batch dimension raises AssertionError."""
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=2)
        dp_metadata = self._create_mock_dp_metadata([5, 7])
        ctx = self._create_mock_forward_context(dp_metadata=dp_metadata)
        mock_get_ctx.return_value = ctx

        mock_ascend_ctx.return_value = ctx
        topk_ids = torch.randint(0, 8, (100, 2)).to(torch.int32)

        with pytest.raises(AssertionError, match="unexpected topk_ids batch"):
            capture(capturer, layer_id=0, topk_ids=topk_ids)

    @patch("vllm_ascend.patch.worker.patch_routed_experts_capture.get_forward_context")
    def test_layer_id_out_of_bounds(self, mock_get_ctx):
        """Reject an incomplete replay rather than silently dropping a layer."""
        capturer = self._create_mock_capturer(dp_rank=0, tp_size=1)
        ctx = self._create_mock_forward_context(dp_metadata=None)
        mock_get_ctx.return_value = ctx

        topk_ids = torch.tensor([[0, 1]], dtype=torch.int32)
        capturer.device_buffer = torch.zeros((100, 5, 2))

        with pytest.raises(IndexError):
            capture(capturer, layer_id=10, topk_ids=topk_ids)

        assert torch.all(capturer.device_buffer == 0)


@pytest.mark.parametrize("raw_rows", [1, 7, 466])
@pytest.mark.parametrize("tp_rank", range(8))
def test_flashcomm_r3_capture_uses_padded_layout_with_dp_metadata(raw_rows, tp_rank):
    from vllm_ascend.ascend_forward_context import MoECommType
    from vllm_ascend.patch.worker import patch_routed_experts_capture as target

    tp_size, topk = 8, 8
    local_rows = (raw_rows + tp_size - 1) // tp_size
    padded_rows = local_rows * tp_size
    ids = torch.arange(padded_rows * topk, dtype=torch.int32).reshape(padded_rows, topk) % 16
    weights = torch.arange(padded_rows * topk, dtype=torch.float32).reshape(padded_rows, topk).to(torch.bfloat16)
    weight_bytes = weights.view(torch.uint8).reshape(padded_rows, topk, 2)
    packed = torch.cat([ids, weight_bytes[..., 0].int(), weight_bytes[..., 1].int()], dim=-1)
    capturer = SimpleNamespace(
        dp_rank=0,
        tp_size=tp_size,
        device_buffer=torch.full((padded_rows, 1, topk * 3), -1, dtype=torch.int32),
    )
    context = SimpleNamespace(dp_metadata=SimpleNamespace(num_tokens_across_dp_cpu=torch.tensor([raw_rows])))
    extra = SimpleNamespace(
        moe_comm_type=MoECommType.ALLTOALL,
        moe_comm_method=SimpleNamespace(
            prepare_finalize=SimpleNamespace(replace_allreduce=True, num_tokens=local_rows)
        ),
        padded_num_tokens=padded_rows,
    )
    begin = tp_rank * local_rows

    def all_gather(outputs, source, group):
        assert all(output.shape == source.shape for output in outputs)
        torch.testing.assert_close(source, packed[begin : begin + local_rows])
        for rank, output in enumerate(outputs):
            output.copy_(packed[rank * local_rows : (rank + 1) * local_rows])

    with (
        patch.object(target, "get_forward_context", return_value=context),
        patch.object(target, "_EXTRA_CTX", extra),
        patch.object(target, "get_tp_group", return_value=SimpleNamespace(device_group=object())),
        patch.object(target.dist, "all_gather", side_effect=all_gather) as gather,
    ):
        target.capture(capturer, 0, ids[begin : begin + local_rows], weights[begin : begin + local_rows])
    gather.assert_called_once()
    torch.testing.assert_close(capturer.device_buffer[:raw_rows, 0], packed[:raw_rows])
    assert (capturer.device_buffer[raw_rows:] == -1).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
