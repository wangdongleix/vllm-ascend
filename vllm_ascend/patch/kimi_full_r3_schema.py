# SPDX-License-Identifier: Apache-2.0
"""Shared Kimi routing schema for worker and scheduler buffers."""

from vllm.model_executor.layers.fused_moe import routed_experts_capturer

KIMI_FULL_R3_PACK_FACTOR = 3
_GET_ROUTED_EXPERTS_SHAPE = routed_experts_capturer._get_routed_experts_shape


def get_kimi_full_r3_shape(vllm_config) -> tuple[int, int, int]:
    """Reserve IDs plus both bytes of each executed BF16 routing weight."""
    layers, experts, topk = _GET_ROUTED_EXPERTS_SHAPE(vllm_config)
    if vllm_config.model_config.hf_config.model_type == "kimi_k3":
        topk *= KIMI_FULL_R3_PACK_FACTOR
    return layers, experts, topk


def install_kimi_full_r3_schema_patch() -> None:
    """Install before either process allocates its routing buffer."""
    routed_experts_capturer._get_routed_experts_shape = get_kimi_full_r3_shape
