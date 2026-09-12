# SPDX-License-Identifier: Apache-2.0
"""Process-wide wire schema for Kimi K3 full routing replay.

The worker captures routes on device while the scheduler owns the persistent
CPU slot buffer.  Both processes must therefore install the same schema before
either buffer is allocated.  Kimi full R3 carries the executed expert IDs and
the two bytes of each executed BF16 router weight in three int32 lanes per
logical top-k entry.
"""

from vllm.model_executor.layers.fused_moe import (
    routed_experts_capturer as _upstream_routed_experts_capturer,
)

from vllm_ascend.transformers_utils.configs.kimi_k3 import KimiK3TextConfig

KIMI_FULL_R3_PACK_FACTOR = 3
_ORIGINAL_GET_NUM_EXPERTS_PER_TOK = _upstream_routed_experts_capturer._get_num_experts_per_tok


def get_kimi_full_r3_wire_width(hf_config) -> int:
    topk = _ORIGINAL_GET_NUM_EXPERTS_PER_TOK(hf_config)
    return topk * KIMI_FULL_R3_PACK_FACTOR if isinstance(hf_config, KimiK3TextConfig) else topk


def install_kimi_full_r3_schema_patch() -> None:
    """Set the shared width helper before worker and scheduler allocation."""
    _upstream_routed_experts_capturer._get_num_experts_per_tok = get_kimi_full_r3_wire_width
