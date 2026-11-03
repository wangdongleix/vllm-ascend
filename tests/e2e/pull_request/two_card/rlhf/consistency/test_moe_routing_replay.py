#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""End-to-end routing replay consistency tests for MoE models.

Only the V2 model runner is exercised: routed-experts capture (R3 / routing
replay) is owned by vLLM's AuxOutput connector, which is only integrated in the
V2 model runner, so the V1 model runner no longer serves R3.
"""

import os
from unittest.mock import patch

import pytest
from vllm import SamplingParams
from vllm.sampling_params import RequestOutputKind

from tests.e2e.conftest import VllmRunner

MODELS = [
    "Qwen/Qwen3.5-35B-A3B",
    "Qwen/Qwen3-30B-A3B",
]


@pytest.mark.parametrize("model", MODELS)
@patch.dict(os.environ, {"OMP_NUM_THREADS": "1", "VLLM_USE_V2_MODEL_RUNNER": "1"})
def test_moe_routing_replay(model):
    """Catch missing TP shards in routing capture (#15451).

    Separate short/long prefills select MC2/ALLTOALL at graph capacity 8.
    Every executed causal row must retain its routing decision on both paths.
    """
    prompts = [
        "Hello, please introduce yourself.",  # 6 tokens -> MC2
        "Hello, please introduce yourself. " * 8,  # 49 tokens -> ALLTOALL
    ]
    with VllmRunner(
        model,
        tensor_parallel_size=2,
        enable_expert_parallel=True,
        cudagraph_capture_sizes=[1, 2, 4, 8],
        distributed_executor_backend="mp",
        enable_return_routed_experts=True,
        async_scheduling=False,
    ) as vllm_model:
        sampling_params = SamplingParams(
            max_tokens=5,
            temperature=0.8,
            top_p=0.95,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )
        for prompt in prompts:
            inputs = vllm_model.get_inputs(prompts=[prompt])
            (request,) = vllm_model.model.generate(prompts=inputs, sampling_params=sampling_params)
            (completion,) = request.outputs
            assert request.finished and completion.text
            routed_experts = completion.routed_experts
            assert routed_experts is not None
            # The last sampled token has not been forwarded yet, so it has no routing.
            expected_rows = len(request.prompt_token_ids) + len(completion.token_ids) - 1
            assert routed_experts.shape[0] == expected_rows
            empty_rows = [index for index, row in enumerate(routed_experts) if not row.any()]
            assert not empty_rows, (
                f"rows {empty_rows} of {expected_rows} carry no routing data for a "
                f"{len(request.prompt_token_ids)}-token prompt"
            )
