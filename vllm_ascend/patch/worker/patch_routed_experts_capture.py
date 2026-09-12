# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/routed_experts_capturer.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
"""Ascend worker capture for complete Kimi full-model R3 payloads."""

from __future__ import annotations

import torch
import torch.distributed as dist
from vllm.distributed.parallel_state import get_tp_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fused_moe import (
    routed_experts_capturer as _upstream_routed_experts_capturer,
)

from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.patch.kimi_full_r3_schema import (
    KIMI_FULL_R3_PACK_FACTOR,
    install_kimi_full_r3_schema_patch,
)

RoutedExpertsCapturer = _upstream_routed_experts_capturer.RoutedExpertsCapturer

# Direct imports of the worker patch must be just as safe as normal plugin
# startup.  This is idempotent and keeps worker and EngineCore schemas equal.
install_kimi_full_r3_schema_patch()


def _tp_shard_layout(
    *,
    token_num_per_dp: int,
    max_tokens: int,
    local_rows: int,
    tp_size: int,
    moe_comm_type: MoECommType,
) -> int | None:
    """Return gathered row count when the router output is TP-sharded."""
    if tp_size <= 1:
        return None

    if moe_comm_type == MoECommType.ALLTOALL:
        gathered_rows = max(token_num_per_dp, tp_size)
        base, remainder = divmod(gathered_rows, tp_size)
        expected_local_rows = {base}
        if remainder:
            expected_local_rows.add(base + 1)
        return gathered_rows if local_rows in expected_local_rows else None

    if moe_comm_type in {MoECommType.MC2, MoECommType.FUSED_MC2}:
        rows_per_rank = (max_tokens + tp_size - 1) // tp_size
        return rows_per_rank * tp_size if local_rows == rows_per_rank else None

    # ALLGATHER reconstructs the token dimension before expert selection.
    return None


def _gather_tp_shards(
    payload: torch.Tensor,
    *,
    gathered_rows: int,
    tp_size: int,
) -> torch.Tensor:
    gathered = torch.empty(
        (gathered_rows, payload.shape[1]),
        dtype=payload.dtype,
        device=payload.device,
    )
    shards = torch.tensor_split(gathered, tp_size, dim=0)
    dist.all_gather(list(shards), payload, get_tp_group().device_group)
    return gathered


def _pack_kimi_full_r3(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Pack executed IDs and exact BF16 weight bits into integer lanes."""
    if topk_weights.shape != topk_ids.shape:
        raise AssertionError(
            "Kimi full R3 requires identical id/weight shapes, got "
            f"ids={tuple(topk_ids.shape)}, weights={tuple(topk_weights.shape)}"
        )
    if topk_weights.dtype != torch.bfloat16:
        raise TypeError(
            f"Kimi full R3 wire schema is bit-exact only for executed BF16 router weights, got {topk_weights.dtype}"
        )

    # Ascend hosts are little-endian, so view(uint8) yields low/high bytes.
    weight_bytes = topk_weights.contiguous().view(torch.uint8).reshape(*topk_weights.shape, 2)
    return torch.cat(
        (
            topk_ids.to(torch.int32),
            weight_bytes[..., 0].to(torch.int32),
            weight_bytes[..., 1].to(torch.int32),
        ),
        dim=-1,
    )


def capture(
    self,
    layer_id: int,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor | None = None,
) -> None:
    """Capture the exact expert/weight pairs consumed by Ascend FusedMoE."""
    actual_topk = int(topk_ids.shape[-1])
    capture_width = int(self.device_buffer.shape[-1])
    if capture_width == actual_topk * KIMI_FULL_R3_PACK_FACTOR:
        if topk_weights is None:
            raise RuntimeError(
                "Kimi full R3 capture buffer is active but FusedMoE did not provide executed router weights"
            )
        routing_payload = _pack_kimi_full_r3(topk_ids, topk_weights)
    elif capture_width == actual_topk:
        routing_payload = topk_ids.to(torch.int32)
    else:
        raise AssertionError(
            "RoutedExpertsCapturer buffer width is incompatible with router "
            f"output: buffer={capture_width}, topk={actual_topk}"
        )

    ctx = get_forward_context()
    if ctx.dp_metadata is None:
        # ctx.num_tokens is the full padded model-input length. Ascend's
        # ALLTOALL/MC2 prepare path may already have split the routed tensor
        # across TP ranks before expert selection, even with a single DP
        # rank. Inferring the full row count from routing_payload would
        # therefore copy only one TP shard and leave the remaining capture
        # rows as zero/expert-0 placeholders.
        token_num_per_dp = int(ctx.num_tokens)
        max_tokens = token_num_per_dp
        local_rows = int(routing_payload.shape[0])
        start_loc = 0
        end_loc = token_num_per_dp
        if local_rows != token_num_per_dp:
            gathered_rows = _tp_shard_layout(
                token_num_per_dp=token_num_per_dp,
                max_tokens=max_tokens,
                local_rows=local_rows,
                tp_size=self.tp_size,
                moe_comm_type=_EXTRA_CTX.moe_comm_type,
            )
            if gathered_rows is None:
                raise AssertionError(
                    "RoutedExpertsCapturer: unexpected single-DP payload "
                    f"batch dim {local_rows} (full={token_num_per_dp}, "
                    f"tp_size={self.tp_size}, "
                    f"moe_comm_type={_EXTRA_CTX.moe_comm_type})"
                )
            routing_payload = _gather_tp_shards(
                routing_payload,
                gathered_rows=gathered_rows,
                tp_size=self.tp_size,
            )
    else:
        num_tokens_dp = ctx.dp_metadata.num_tokens_across_dp_cpu
        token_num_per_dp = int(num_tokens_dp[self.dp_rank].item())
        total = int(num_tokens_dp.sum().item())
        n = routing_payload.shape[0]
        max_tokens = int(num_tokens_dp.max().item())
        total_with_padding = max_tokens * len(num_tokens_dp)

        if n == total:
            # Naive dispatch: all DP ranks' tokens concatenated
            # before routing. This rank owns tokens
            # [end_loc - token_num_per_dp, end_loc).
            cumsum = torch.cumsum(num_tokens_dp, dim=0)
            end_loc = int(cumsum[self.dp_rank].item())
            start_loc = end_loc - token_num_per_dp
        elif n == token_num_per_dp:
            # Modular-kernel path: DP combine happens inside
            # quant_method.apply; select_experts only sees this
            # rank's tokens, take the whole tensor.
            start_loc = 0
            end_loc = token_num_per_dp
        elif n == total_with_padding:
            # NOTE(Ronald1995): When all DP ranks have equal token counts,
            # total == total_with_padding, so the first branch (n == total)
            # fires instead. This overlap is intentional since both branches
            # produce equivalent results in that case.

            # Padded all-gather path: tokens are padded to max_tokens before
            # all-gather across DP group. Each DP rank occupies a contiguous
            # block of size max_tokens. Extract only the actual tokens for
            # this rank (skip padding).
            # Example: dp_rank=0, max_tokens=7, token_num_per_dp=5.
            # start_loc = 0 * 7 = 0
            # end_loc = 0 + 5 = 5 (only first 5 tokens are valid)

            start_loc = self.dp_rank * max_tokens
            end_loc = start_loc + token_num_per_dp
        elif (
            n != token_num_per_dp
            and (
                gathered_rows := _tp_shard_layout(
                    token_num_per_dp=token_num_per_dp,
                    max_tokens=max_tokens,
                    local_rows=n,
                    tp_size=self.tp_size,
                    moe_comm_type=_EXTRA_CTX.moe_comm_type,
                )
            )
            is not None
        ):
            routing_payload = _gather_tp_shards(
                routing_payload,
                gathered_rows=gathered_rows,
                tp_size=self.tp_size,
            )
            start_loc = 0
            end_loc = token_num_per_dp
        else:
            sp_expected = (token_num_per_dp + self.tp_size - 1) // self.tp_size if self.tp_size > 0 else -1
            raise AssertionError(
                "RoutedExpertsCapturer: unexpected payload batch dim "
                f"{n} (expected {total}, {token_num_per_dp}, "
                f"{total_with_padding}, or {sp_expected}; "
                f"dp_rank={self.dp_rank}, tp_size={self.tp_size}, "
                f"moe_comm_type={_EXTRA_CTX.moe_comm_type})"
            )

    if layer_id < 0 or layer_id >= self.device_buffer.shape[1]:
        raise IndexError(
            "RoutedExpertsCapturer layer_id is outside capture buffer: "
            f"layer_id={layer_id}, layers={self.device_buffer.shape[1]}"
        )

    selected = routing_payload[start_loc:end_loc, :]
    if selected.shape != (token_num_per_dp, capture_width):
        raise RuntimeError(
            "Kimi full R3 capture slice is incomplete: "
            f"selected={tuple(selected.shape)}, expected="
            f"({token_num_per_dp}, {capture_width})"
        )
    self.device_buffer[:token_num_per_dp, layer_id, :] = selected



RoutedExpertsCapturer.capture = capture
