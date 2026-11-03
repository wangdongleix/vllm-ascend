# Supported Features

The feature support principle of vLLM Ascend is: **aligned with vLLM**. We are also actively collaborating with the community to accelerate support.

Tool calling: [vLLM documentation](https://docs.vllm.ai/en/latest/features/tool_calling/)

You can check the [support status of vLLM V1 Engine][v1_user_guide]. Below is the feature support status of vLLM Ascend:

| Feature                       |      Status    | Next Step                                                              |
|-------------------------------|----------------|------------------------------------------------------------------------|
| Chunked Prefill               | 🟢 Functional    | Functional, see detailed note: [Chunked Prefill][cp]                     |
| Automatic Prefix Caching      | 🟢 Functional    | Functional, see detailed note: [vllm-ascend#732][apc]                    |
| LoRA                          | 🔵 Experimental  | Functional, see detailed note: [LoRA][LoRA]                    |
| Speculative decoding          | 🟢 Functional    | Basic support                                                          |
| Pooling                       | 🔵 Experimental  | CI needed to adapt to more models; V1 support relies on vLLM support.   |
| Enc-dec                       | 🟡 Planned       | vLLM should support this feature first.                                |
| Multi Modality                | 🟢 Functional    | [Multi Modality][multimodal], optimizing and adapting more models            |
| LogProbs                      | 🟢 Functional    | CI needed                                                              |
| Prompt LogProbs               | 🟢 Functional    | CI needed                                                              |
| Async output                  | 🟢 Functional    | CI needed                                                              |
| Beam search                   | 🔵 Experimental  | CI needed                                                              |
| Guided Decoding               | 🟢 Functional    | [vllm-ascend#177][guided_decoding]                                     |
| Tensor Parallel               | 🟢 Functional    | Make TP >4 work with graph mode.                                        |
| Pipeline Parallel             | 🟢 Functional    | See [Pipeline Parallelism][pipeline_parallel]. An end-to-end tutorial is planned. |
| Expert Parallelism            | 🟢 Functional    | Supports MRv1 legacy EPLB and MRv2 asynchronous EPLB with the Ascend STAIR policy as the default; see [EPLB][eplb]. |
| Data Parallel                 | 🟢 Functional    | Data Parallel support for Qwen3 MoE.                                   |
| Prefill Decode Disaggregation | 🟢 Functional    | Functional, xPyD is supported.                                         |
| Quantization                  | 🟢 Functional    | W8A8 available; working on more quantization method support (W4A8, etc.) |
| Graph Mode                    | 🟢 Functional    | Functional, see detailed note: [Graph Mode][graph_mode]                 |
| Sleep Mode                    | 🟢 Functional    | Functional, see detailed note: [Sleep Mode][sleep_mode]                 |
| Context Parallel              | 🟢 Functional    | Functional, see detailed note: [Context Parallel][context_parallel]     |
| KVPP                          | 🔵 Experimental  | Non-hybrid MLA/SFA models in eager mode; see [KVPP][kvpp] for supported combinations. |
| Fine-grained TP               | 🔵 Experimental  | Functional, see detailed note: [Fine-grained TP Doc][finegrained_tp] |
| CPU Binding                   | 🟢 Functional    | Functional, see detailed note: [CPU Binding Doc][cpu_binding] |
| KV Pool                       | 🔵 Experimental  | Functional, see detailed note: [KV Pool Doc][kv_pool] |

- 🟢 Functional: Fully operational, with ongoing optimizations.
- 🔵 Experimental: Experimental support, interfaces and functions may change.
- 🚧 WIP: Under active development, will be supported soon.
- 🟡 Planned: Scheduled for future implementation (some may have open PRs/RFCs).
- 🔴 NO plan/Deprecated: No plan or deprecated by vLLM.

With `enable_return_routed_experts=True`, routing replay uses the upstream
`routed_experts` tensor of logical expert IDs, shaped as
`[token_rows, layers, top_k]`, including for Kimi K3. The training actor computes
its router weights from its own logits. TP/DP capture preserves the complete
token order before expert load-balancing maps logical IDs to physical slots.

The Kimi K3 multimodal processor accepts both the native image placeholder and
media blocks emitted by the checkpoint's HF processor through vLLM's prompt
updates. With the same default image configuration, an HF prompt already carrying
one `media_pad` token per visual feature keeps its token IDs and sequence length;
vLLM records the existing embedding slots without expanding the image again.
Images use the standard PIL `image` modality.

The vLLM 0.30 Kimi image wrapper does not forward per-request image resize
options to the checkpoint image processor. Ascend rejects those grid-changing
overrides, including nonempty `images_kwargs`, instead of silently ignoring them.
Empty `images_kwargs` and ordinary tokenizer/framework options remain accepted.
Per-call image resizing supported by a training processor does not imply support
for that override in the rollout processor; rollout uses its checkpoint image
configuration.

[v1_user_guide]: https://docs.vllm.ai/en/latest/usage/v1_guide/
[multimodal]: https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen-VL-Dense.html
[guided_decoding]: https://github.com/vllm-project/vllm-ascend/issues/177
[LoRA]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/lora.html
[graph_mode]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/graph_mode.html
[apc]: https://github.com/vllm-project/vllm-ascend/issues/732
[cp]: https://docs.vllm.ai/en/stable/configuration/optimization/
[1P1D]: https://github.com/vllm-project/vllm-ascend/pull/950
[context_parallel]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/context_parallel.html
[sleep_mode]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/sleep_mode.html
[eplb]: https://docs.vllm.ai/projects/ascend/en/latest/user_guide/feature_guide/expert_parallelism_load_balancer.html
[pipeline_parallel]: ../feature_guide/pipeline_parallel.md
[kvpp]: ../feature_guide/kvpp.md
[finegrained_tp]: ../feature_guide/Fine_grained_TP.md
[cpu_binding]: ../feature_guide/cpu_binding.md
[kv_pool]: ../feature_guide/kv_cache_pool_ascend_store.md
