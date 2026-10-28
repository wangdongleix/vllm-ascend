# SPDX-License-Identifier: Apache-2.0
"""Preserve the executed Kimi routing weights for training replay."""

import torch
from vllm.model_executor.layers.fused_moe.router.base_router import BaseRouter


class KimiTrainingRouter(BaseRouter):
    """Reuse Ascend selection and capture logical IDs with executed weights."""

    def __init__(self, router: BaseRouter) -> None:
        super().__init__(router.top_k, router.global_num_experts, router.eplb_state)
        self.router = router

    @property
    def routing_method_type(self):
        return self.router.routing_method_type

    def _compute_routing(self, hidden_states, router_logits, indices_type, *, input_ids=None):
        return self.router._compute_routing(hidden_states, router_logits, indices_type, input_ids=input_ids)

    def _select_experts(self, hidden_states, router_logits, topk_indices_dtype=None, *, input_ids=None):
        self._validate_eplb_state()
        weights, ids = self._compute_routing(hidden_states, router_logits, topk_indices_dtype, input_ids=input_ids)
        weights = weights.to(torch.bfloat16)
        if self.capture_fn is not None:
            self.capture_fn(ids, weights)
        ids = self._apply_eplb_mapping(ids)
        return weights, self._convert_indices_dtype(ids, topk_indices_dtype)
