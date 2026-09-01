# SPDX-License-Identifier: Apache-2.0
"""Process-wide wire schema for Kimi K3 full routing replay.

The worker captures routes on device while the scheduler owns the persistent
CPU slot buffer.  Both processes must therefore install the same schema before
either buffer is allocated.  Kimi full R3 carries the executed expert IDs and
the two bytes of each executed BF16 router weight in three int32 lanes per
logical top-k entry.
"""

from __future__ import annotations

import functools
from typing import Any

from vllm.model_executor.layers.fused_moe import (
    routed_experts_capturer as _upstream_routed_experts_capturer,
)

KIMI_FULL_R3_PACK_FACTOR = 3
_KIMI_MODEL_TYPES = frozenset({"kimi_k3", "kimi_linear"})
_ORIGINAL_GET_NUM_EXPERTS_PER_TOK = (
    _upstream_routed_experts_capturer._get_num_experts_per_tok
)


def _config_candidates(hf_config: Any):
    yield hf_config
    for name in ("text_config", "language_config", "llm_config"):
        nested = getattr(hf_config, name, None)
        if nested is not None:
            yield nested


def is_kimi_full_r3_config(hf_config: Any) -> bool:
    return any(
        str(getattr(candidate, "model_type", "")).lower()
        in _KIMI_MODEL_TYPES
        for candidate in _config_candidates(hf_config)
    )


def get_logical_num_experts_per_tok(hf_config: Any) -> int:
    """Return the real router top-k before full-R3 wire packing."""
    for candidate in _config_candidates(hf_config):
        for name in (
            "num_experts_per_token",
            "num_experts_per_tok",
            "top_k_experts",
        ):
            value = getattr(candidate, name, None)
            if value is not None:
                value = int(value)
                if value <= 0:
                    raise ValueError(f"invalid routed top-k width: {value}")
                return value

    value = int(_ORIGINAL_GET_NUM_EXPERTS_PER_TOK(hf_config))
    if value <= 0:
        raise ValueError(f"invalid routed top-k width: {value}")
    return value


def get_kimi_full_r3_wire_width(hf_config: Any) -> int:
    if not is_kimi_full_r3_config(hf_config):
        return int(_ORIGINAL_GET_NUM_EXPERTS_PER_TOK(hf_config))
    return get_logical_num_experts_per_tok(hf_config) * KIMI_FULL_R3_PACK_FACTOR


def _patch_scheduler_manager() -> None:
    manager_cls = _upstream_routed_experts_capturer.RoutedExpertsManager
    if getattr(manager_cls, "_kimi_full_r3_schema_patch", False):
        return

    original_init = manager_cls.__init__

    @functools.wraps(original_init)
    def full_r3_init(self, vllm_config, kv_cache_config) -> None:
        hf_config = vllm_config.model_config.hf_text_config
        kimi_full_r3 = is_kimi_full_r3_config(hf_config)
        logical_topk = (
            get_logical_num_experts_per_tok(hf_config)
            if kimi_full_r3
            else None
        )

        # vLLM 0.26 sizes through the helper but its informational log still
        # reads the legacy attribute directly.  Supply the alias only while
        # upstream initializes the manager.
        temporary_topk_alias = False
        if kimi_full_r3 and not hasattr(hf_config, "num_experts_per_tok"):
            setattr(hf_config, "num_experts_per_tok", logical_topk)
            temporary_topk_alias = True
        try:
            original_init(self, vllm_config, kv_cache_config)
        finally:
            if temporary_topk_alias:
                delattr(hf_config, "num_experts_per_tok")

        if not kimi_full_r3:
            self._kimi_full_r3_schema_active = False
            return

        expected_width = int(logical_topk) * KIMI_FULL_R3_PACK_FACTOR
        buffer = self.routed_experts_by_slot
        actual_width = int(buffer.shape[-1])
        if actual_width != expected_width:
            raise RuntimeError(
                "Kimi full R3 scheduler schema mismatch during init: "
                f"logical_topk={logical_topk}, expected_payload_width="
                f"{expected_width}, actual_payload_width={actual_width}, "
                f"buffer_shape={buffer.shape}"
            )

        self._kimi_full_r3_schema_active = True
        marker = (
            "Kimi vLLM full R3 SCHEDULER ACTIVE: "
            f"slots={buffer.shape[0]} layers={buffer.shape[1]} "
            f"logical_topk={logical_topk} payload_width={actual_width} "
            f"dtype={buffer.dtype.name} ids_and_bf16_weights=True"
        )
        print(marker, flush=True)

    full_r3_init.__module__ = __name__
    manager_cls.__init__ = full_r3_init
    manager_cls._kimi_full_r3_schema_patch = True


def install_kimi_full_r3_schema_patch() -> None:
    """Install the shared helper and scheduler guards exactly once."""
    _upstream_routed_experts_capturer._get_num_experts_per_tok = (
        get_kimi_full_r3_wire_width
    )
    _patch_scheduler_manager()
