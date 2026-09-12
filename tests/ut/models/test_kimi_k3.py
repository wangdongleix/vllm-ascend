# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm.model_executor.models.utils import StageMissingLayer

from vllm_ascend.models import kimi_k3
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3ForCausalLM,
    AscendKimiK3ForConditionalGeneration,
    KimiK3MLP,
    KimiK3MoE,
    KimiK3MultiModalProjector,
    KimiK3TextModel,
    KimiK3VisionEncoderLayer,
    _move_module_to_device,
    _resolve_packed_expert_weight_name,
    _routed_latent_quant_config,
    get_spec_layer_idx_from_weight_name,
)
from vllm_ascend.ops.activation import AscendSituAndMul, SituActivationConfig
from vllm_ascend.transformers_utils.configs.kimi_k3 import (
    KimiK3Config,
    KimiK3VisionConfig,
)


def test_kimi_k3_model_declares_checkpoint_packing_contract():
    assert AscendKimiK3ForCausalLM.packed_modules_mapping["fused_qkv"] == [
        "q_proj",
        "k_proj",
        "v_proj",
    ]
    assert AscendKimiK3ForCausalLM.packed_modules_mapping["experts"] == [
        "experts.0.w1",
        "experts.0.w3",
        "experts.0.w2",
    ]


def test_kimi_k3_loads_qkv_checkpoint_shards_into_separate_linears():
    model = KimiK3TextModel.__new__(KimiK3TextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=0)
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].self_attn = nn.Module()
    projection_weights = {}
    for name in ("q_proj", "k_proj", "v_proj"):
        projection = nn.Module()
        weight = nn.Parameter(torch.empty(1))
        weight.weight_loader = MagicMock()
        projection.register_parameter("weight", weight)
        setattr(model.layers[0].self_attn, name, projection)
        projection_weights[name] = weight
    weights = [(f"layers.0.self_attn.{name}.weight", torch.empty(1)) for name in ("q_proj", "k_proj", "v_proj")]

    with (
        patch("vllm_ascend.models.kimi_k3.get_spec_layer_idx_from_weight_name", return_value=None),
        patch("vllm_ascend.models.kimi_k3.fused_moe_make_expert_params_mapping", return_value=[]),
        patch("vllm_ascend.models.kimi_k3.is_pp_missing_parameter", return_value=False),
    ):
        loaded = model.load_weights(weights)

    for name, weight in projection_weights.items():
        weight.weight_loader.assert_called_once()
        assert weight.weight_loader.call_args.args[1] is weights[("q_proj", "k_proj", "v_proj").index(name)][1]
    assert loaded == {
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
    }


@pytest.mark.parametrize(
    ("quant_name", "uses_quantized_latent_projections"),
    [
        ("ascend", True),
        ("compressed-tensors", False),
        ("other", False),
    ],
)
def test_kimi_k3_quantizes_latent_projections_only_for_modelslim(
    quant_name: str,
    uses_quantized_latent_projections: bool,
):
    quant_config = MagicMock()
    quant_config.get_name.return_value = quant_name

    actual = _routed_latent_quant_config(quant_config)

    if uses_quantized_latent_projections:
        assert actual is quant_config
    else:
        assert actual is None


def test_kimi_k3_unquantized_model_keeps_latent_projections_unquantized():
    assert _routed_latent_quant_config(None) is None


def test_kimi_k3_projector_registers_rotation_for_weight_loading(
    monkeypatch: pytest.MonkeyPatch,
):
    class StubReplicatedLinear(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def forward(self, hidden_states):
            return hidden_states, None

    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", StubReplicatedLinear)
    monkeypatch.setattr(kimi_k3, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(kimi_k3, "get_act_fn", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(
        kimi_k3,
        "is_vit_use_data_parallel",
        lambda num_heads: False,
    )
    config = KimiK3VisionConfig(
        mm_hidden_size=2,
        text_hidden_size=8,
        merge_kernel_size=(2, 2),
        use_rot_proj=True,
    )
    projector = KimiK3MultiModalProjector(config)

    assert projector.rot_proj is not None


@pytest.mark.parametrize("wrapped", [False, True])
def test_kimi_k3_bucketed_reload_preserves_configured_rotation(monkeypatch, wrapped):
    loaded_weights = {"mm_projector.linear_1.weight"}

    class StubLoader:
        def __init__(self, model):
            assert model is wrapper

        def load_weights(self, weights, *, mapper):
            return loaded_weights

    monkeypatch.setattr(kimi_k3, "AutoWeightsLoader", StubLoader)
    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    projector = nn.Module()
    rotation = nn.Linear(1, 1, bias=False)
    projector.rot_proj = rotation
    wrapper.mm_projector = StageMissingLayer("vision_tower", projector) if wrapped else projector
    for _ in range(2):
        assert wrapper.load_weights(iter(())) == loaded_weights
        assert projector.rot_proj is rotation


def test_kimi_k3_projector_applies_rotation_only_after_weight_load():
    class PassthroughLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    class ScaleLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states * 2, None

    projector = KimiK3MultiModalProjector.__new__(KimiK3MultiModalProjector)
    nn.Module.__init__(projector)
    projector.input_size = 2
    projector.use_native_linear = False
    projector.linear_1 = PassthroughLinear()
    projector.linear_2 = PassthroughLinear()
    projector.act = nn.Identity()
    projector.post_norm = nn.Identity()
    image_features = torch.tensor([[1.0, 2.0]])

    projector.rot_proj = ScaleLinear()
    projector.rot_proj = None
    assert projector.rot_proj is None
    torch.testing.assert_close(projector(image_features), image_features)

    projector.rot_proj = ScaleLinear()
    torch.testing.assert_close(projector(image_features), image_features * 2)


@pytest.mark.parametrize(
    ("name", "params", "expected"),
    [
        (
            "layers.1.experts.w13_weight",
            {"layers.1.experts.w13_weight": object()},
            "layers.1.experts.w13_weight",
        ),
        (
            "layers.1.experts.w13_weight",
            {"layers.1.experts.w13_weight_packed": object()},
            "layers.1.experts.w13_weight_packed",
        ),
        (
            "layers.1.experts.w2_weight",
            {"layers.1.experts.w2_weight_packed": object()},
            "layers.1.experts.w2_weight_packed",
        ),
        (
            "layers.1.experts.w13_weight",
            {"layers.1.experts.routed_experts.w13_weight": object()},
            "layers.1.experts.routed_experts.w13_weight",
        ),
        (
            "layers.1.experts.w2_weight",
            {"layers.1.experts.routed_experts.w2_weight_packed": object()},
            "layers.1.experts.routed_experts.w2_weight_packed",
        ),
        (
            "layers.1.experts.w13_weight_scale",
            {"layers.1.experts.w13_weight_packed": object()},
            "layers.1.experts.w13_weight_scale",
        ),
    ],
)
def test_kimi_k3_resolves_packed_expert_checkpoint_names(
    name: str,
    params: dict[str, object],
    expected: str,
):
    assert _resolve_packed_expert_weight_name(name, params) == expected


def test_kimi_k3_loads_verl_packed_local_experts_into_v026_routed_experts():
    model = KimiK3TextModel.__new__(KimiK3TextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=32)
    model.layers = nn.ModuleList([nn.Module(), nn.Module()])
    moe = nn.Module()
    moe.experts = nn.Module()
    moe.experts.routed_experts = nn.Module()
    model.layers[1].block_sparse_moe = moe

    w13_weight = nn.Parameter(torch.empty(1))
    w2_weight = nn.Parameter(torch.empty(1))
    w13_weight.weight_loader = MagicMock(return_value=True)
    w2_weight.weight_loader = MagicMock(return_value=True)
    moe.experts.routed_experts.register_parameter("w13_weight", w13_weight)
    moe.experts.routed_experts.register_parameter("w2_weight", w2_weight)

    gate_up = torch.arange(4 * 3 * 10, dtype=torch.float32).view(4, 3, 10)
    down = torch.arange(4 * 5 * 3, dtype=torch.float32).view(4, 5, 3)
    weights = [
        (
            "layers.1.block_sparse_moe.experts.gate_up_proj.__verl_packed_local__.8",
            gate_up,
        ),
        (
            "layers.1.block_sparse_moe.experts.down_proj.__verl_packed_local__.8",
            down,
        ),
    ]

    with patch(
        "vllm_ascend.models.kimi_k3.fused_moe_make_expert_params_mapping",
        return_value=[],
    ):
        loaded = model.load_weights(weights)

    expected_prefix = "layers.1.block_sparse_moe.experts.routed_experts"
    assert loaded == {
        f"{expected_prefix}.w13_weight",
        f"{expected_prefix}.w2_weight",
    }
    assert [call.kwargs["expert_id"] for call in w13_weight.weight_loader.call_args_list] == [
        8,
        8,
        9,
        9,
        10,
        10,
        11,
        11,
    ]
    assert [call.kwargs["shard_id"] for call in w13_weight.weight_loader.call_args_list] == ["w1", "w3"] * 4
    assert [call.kwargs["expert_id"] for call in w2_weight.weight_loader.call_args_list] == [8, 9, 10, 11]
    torch.testing.assert_close(
        w13_weight.weight_loader.call_args_list[0].args[1],
        gate_up[0, :, :5].t().contiguous(),
    )
    torch.testing.assert_close(
        w13_weight.weight_loader.call_args_list[1].args[1],
        gate_up[0, :, 5:].t().contiguous(),
    )
    torch.testing.assert_close(
        w2_weight.weight_loader.call_args_list[0].args[1],
        down[0].t().contiguous(),
    )


def test_kimi_k3_config_normalizes_checkpoint_schema_for_vllm():
    """Cover only the non-pass-through checkpoint-to-vLLM adaptations."""
    config = KimiK3Config(
        text_config={"hidden_size": 4096},
        vision_config={
            "vt_num_attention_heads": 12,
            "vt_num_hidden_layers": 7,
            "vt_hidden_size": 1024,
            "vt_intermediate_size": 3584,
            "text_hidden_size": 1024,
        },
        use_unified_vision_chunk=True,
    )

    # vLLM's renderer must normalize standard image content to the same
    # vision_chunk modality consumed by verl's Kimi adapter.
    assert config.use_unified_vision_chunk is True
    # MoonViT consumers use canonical names instead of checkpoint vt_* names.
    assert config.vision_config.num_attention_heads == 12
    assert config.vision_config.hidden_size == 1024
    # The projector output must follow the text model, not stale vision config.
    assert config.vision_config.text_hidden_size == config.text_config.hidden_size


def test_kimi_k3_model_uses_unified_vision_chunk_placeholder():
    assert AscendKimiK3ForConditionalGeneration.get_placeholder_str("image", 0) == ("<|media_begin|>image<|media_content|><|media_pad|><|media_end|>")
    with pytest.raises(ValueError, match="does not support modality"):
        AscendKimiK3ForConditionalGeneration.get_placeholder_str(
            "vision_chunk",
            0,
        )


def test_kimi_k3_weight_mapper_adds_inner_language_model_prefix():
    mapper = AscendKimiK3ForConditionalGeneration.hf_to_vllm_mapper

    assert (
        mapper._map_name("language_model.layers.12.self_attn.q_proj.weight")
        == "language_model.model.layers.12.self_attn.q_proj.weight"
    )
    assert (
        mapper._map_name("language_model.model.layers.12.self_attn.q_proj.weight")
        == "language_model.model.layers.12.self_attn.q_proj.weight"
    )
    assert mapper._map_name("mm_projector.proj.0.weight") == "mm_projector.linear_1.weight"


@pytest.mark.parametrize(
    ("weight_name", "expected_layer"),
    [
        ("model.layers.93.self_attn.q_proj.weight", 93),
        ("layers.94.self_attn.q_proj.weight", 94),
        ("language_model.layers.93.mlp.gate_proj.weight", 93),
        ("language_model.model.layers.94.mlp.up_proj.weight", 94),
        ("model.layers.92.self_attn.q_proj.weight", None),
        ("model.layers.95.self_attn.q_proj.weight", None),
    ],
)
def test_kimi_k3_spec_layer_detection_accepts_loader_prefixes(
    weight_name: str,
    expected_layer: int | None,
):
    config = SimpleNamespace(
        num_hidden_layers=93,
        num_nextn_predict_layers=2,
    )

    assert get_spec_layer_idx_from_weight_name(config, weight_name) == expected_layer


def test_kimi_k3_spec_layer_detection_allows_missing_nextn_config():
    config = SimpleNamespace(num_hidden_layers=93)

    assert (
        get_spec_layer_idx_from_weight_name(
            config,
            "model.layers.93.self_attn.q_proj.weight",
        )
        is None
    )


def test_kimi_k3_vision_tp16_falls_back_to_data_parallel(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.models import vision as vision_utils

    class StubModule(nn.Module):
        pass

    qkv_kwargs: dict[str, object] = {}
    output_kwargs: dict[str, object] = {}

    def fake_qkv(*args, **kwargs):
        del args
        qkv_kwargs.update(kwargs)
        return StubModule()

    def fake_output(*args, **kwargs):
        del args
        output_kwargs.update(kwargs)
        return StubModule()

    monkeypatch.setattr(
        vision_utils,
        "get_tensor_model_parallel_world_size",
        lambda: 16,
    )
    monkeypatch.setattr(
        kimi_k3,
        "get_tensor_model_parallel_world_size",
        lambda: 16,
    )
    monkeypatch.setattr(
        kimi_k3,
        "KimiK3VisionMLP",
        lambda *args, **kwargs: StubModule(),
    )
    monkeypatch.setattr(kimi_k3, "get_act_fn", lambda name: nn.Identity())
    monkeypatch.setattr(kimi_k3, "QKVParallelLinear", fake_qkv)
    monkeypatch.setattr(kimi_k3, "RowParallelLinear", fake_output)
    monkeypatch.setattr(
        kimi_k3,
        "MMEncoderAttention",
        lambda *args, **kwargs: StubModule(),
    )

    layer = KimiK3VisionEncoderLayer(
        KimiK3VisionConfig(vt_num_attention_heads=12),
        quant_config=None,
        prefix="vision_tower.encoder.blocks.0",
    )

    assert layer.use_data_parallel is True
    assert layer.tp_size == 1
    assert layer.num_local_heads == 12
    assert layer.use_native_linear is True
    assert isinstance(layer.wqkv, nn.Linear)
    assert isinstance(layer.wo, nn.Linear)
    # vLLM 0.26's data-parallel vision path uses ordinary torch linears;
    # sharded vLLM linears (and their legacy disable_tp kwarg) are bypassed.
    assert qkv_kwargs == {}
    assert output_kwargs == {}


def test_kimi_k3_skips_explicit_move_for_meta_modules():
    module = nn.Linear(4, 4, device="meta")

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.is_meta for parameter in module.parameters())


def test_kimi_k3_moves_non_meta_modules():
    module = nn.Linear(4, 4)

    actual = _move_module_to_device(
        module,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert actual is module
    assert all(parameter.device.type == "cpu" for parameter in module.parameters())
    assert all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())


def test_kimi_k3_passes_situ_parameters_through_activation_config(monkeypatch):
    class StubModule(nn.Module):
        pass

    fused_moe_kwargs = {}

    def fake_replicated_linear(*args, **kwargs):
        return StubModule()

    def fake_fused_moe(**kwargs):
        fused_moe_kwargs.update(kwargs)
        return StubModule()

    monkeypatch.setattr(kimi_k3, "ReplicatedLinear", fake_replicated_linear)
    monkeypatch.setattr(kimi_k3, "FusedMoE", fake_fused_moe)
    config = SimpleNamespace(
        hidden_act="situ",
        hidden_size=32,
        routed_expert_hidden_size=16,
        num_shared_experts=0,
        num_experts=8,
        rms_norm_eps=1e-6,
        latent_moe_use_norm=False,
        moe_intermediate_size=12,
        num_experts_per_token=2,
        moe_renormalize=True,
        use_grouped_topk=True,
        num_expert_group=4,
        topk_group=2,
        moe_router_activation_func="sigmoid",
        routed_scaling_factor=2.5,
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    KimiK3MoE(config, prefix="model.layers.1.block_sparse_moe")

    activation = fused_moe_kwargs["activation"]
    assert isinstance(activation, SituActivationConfig)
    assert activation.beta == 4.0
    assert activation.linear_beta == 25.0


def test_kimi_k3_dense_mlp_uses_callable_situ(monkeypatch):
    class StubLinear(nn.Module):
        def forward(self, hidden_states):
            return hidden_states, None

    monkeypatch.setattr(kimi_k3, "MergedColumnParallelLinear", lambda *args, **kwargs: StubLinear())
    monkeypatch.setattr(kimi_k3, "RowParallelLinear", lambda *args, **kwargs: StubLinear())
    config = SimpleNamespace(
        hidden_act="situ",
        activation_situ_beta=4.0,
        activation_situ_linear_beta=25.0,
    )

    mlp = KimiK3MLP(config, hidden_size=4, intermediate_size=2)
    hidden_states = torch.tensor([[1.0, -2.0, 3.0, -4.0]])
    output = mlp(hidden_states)

    assert isinstance(mlp.act_fn, AscendSituAndMul)
    assert output.shape == (1, 2)


def test_text_model_captures_materialized_dspark_aux_stream(monkeypatch: pytest.MonkeyPatch):
    residual_calls: list[tuple[int, torch.Tensor]] = []
    consumed_inputs: list[torch.Tensor] = []

    class Marker(nn.Module):
        def __init__(self, value: int) -> None:
            super().__init__()
            self.value = value

    class FakeLayer(nn.Module):
        def __init__(self, layer_idx: int) -> None:
            super().__init__()
            self.layer_idx = layer_idx
            self.self_attention_res_proj = Marker(layer_idx)
            self.self_attention_res_norm = nn.Identity()

        def forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            block_residual: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del positions
            if block_residual.shape[1] > 0:
                hidden_states = kimi_k3._apply_attention_residual(
                    hidden_states,
                    block_residual,
                    self.self_attention_res_proj,
                    self.self_attention_res_norm,
                )
            consumed_inputs.append(hidden_states.clone())
            if block_residual.shape[1] == 0:
                block_residual = hidden_states.unsqueeze(1)
            return hidden_states + 10, block_residual

    def fake_attention_residual(
        hidden_states: torch.Tensor,
        block_residual: torch.Tensor,
        projection: Marker,
        norm: nn.Module,
    ) -> torch.Tensor:
        del block_residual, norm
        residual_calls.append((projection.value, hidden_states.clone()))
        return hidden_states + projection.value * 100

    monkeypatch.setattr(kimi_k3, "_apply_attention_residual", fake_attention_residual)
    monkeypatch.setattr(
        kimi_k3,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )

    model = KimiK3TextModel.__new__(KimiK3TextModel)
    nn.Module.__init__(model)
    model.do_not_compile = True
    model.start_layer = 0
    model.end_layer = 2
    model.layers = nn.ModuleList([FakeLayer(0), FakeLayer(1)])
    model.embed_input_ids = MagicMock(return_value=torch.tensor([[1.0]]))
    model.output_attn_res_proj = Marker(0)
    model.output_attn_res_norm = nn.Identity()
    model.norm = nn.Identity()
    model.dspark_aux_capture_materialized = True
    model._set_aux_hidden_state_layers((0, 1))

    hidden_states, aux_hidden_states = model(
        torch.tensor([1]),
        torch.tensor([0]),
        None,
    )

    torch.testing.assert_close(aux_hidden_states[0], consumed_inputs[0])
    torch.testing.assert_close(aux_hidden_states[1], consumed_inputs[1])
    torch.testing.assert_close(aux_hidden_states[0], torch.tensor([[1.0]]))
    torch.testing.assert_close(aux_hidden_states[1], torch.tensor([[111.0]]))
    torch.testing.assert_close(hidden_states, torch.tensor([[121.0]]))
    assert [layer_idx for layer_idx, _ in residual_calls] == [1, 1, 0]

    residual_calls.clear()
    consumed_inputs.clear()
    model.dspark_aux_capture_materialized = False
    model._set_aux_hidden_state_layers((1,))

    _, raw_aux_hidden_states = model(
        torch.tensor([1]),
        torch.tensor([0]),
        None,
    )

    torch.testing.assert_close(raw_aux_hidden_states[0], torch.tensor([[11.0]]))
    assert [layer_idx for layer_idx, _ in residual_calls] == [1, 0]


def test_kimi_k3_dspark_aux_capture_mode_is_forwarded():
    causal_model = AscendKimiK3ForCausalLM.__new__(AscendKimiK3ForCausalLM)
    nn.Module.__init__(causal_model)
    causal_model.model = SimpleNamespace(dspark_aux_capture_materialized=False)

    causal_model.set_dspark_aux_capture_materialized(True)

    assert causal_model.model.dspark_aux_capture_materialized is True

    wrapper = AscendKimiK3ForConditionalGeneration.__new__(AscendKimiK3ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.language_model = MagicMock()

    wrapper.set_dspark_aux_capture_materialized(True)

    wrapper.language_model.set_dspark_aux_capture_materialized.assert_called_once_with(True)


def test_kimi_k3_loads_quantized_qkv_into_fused_projection():
    model = KimiK3TextModel.__new__(KimiK3TextModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_experts=0)
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].self_attn = nn.Module()
    model.layers[0].self_attn.fused_qkv = nn.Module()
    weight = nn.Parameter(torch.empty(3))
    weight.weight_loader = MagicMock()
    model.layers[0].self_attn.fused_qkv.register_parameter("weight", weight)
    values = [torch.tensor([float(i)]) for i in range(3)]
    weights = [(f"layers.0.self_attn.{name}_proj.weight", value) for name, value in zip("qkv", values)]
    with (
        patch("vllm_ascend.models.kimi_k3.get_spec_layer_idx_from_weight_name", return_value=None),
        patch("vllm_ascend.models.kimi_k3.fused_moe_make_expert_params_mapping", return_value=[]),
        patch("vllm_ascend.models.kimi_k3.is_pp_missing_parameter", return_value=False),
    ):
        loaded = model.load_weights(weights)
    assert loaded == {"layers.0.self_attn.fused_qkv.weight"}
    assert weight.weight_loader.call_count == 3
    for call, value, shard in zip(weight.weight_loader.call_args_list, values, "qkv"):
        assert call.args[0] is weight
        assert call.args[1] is value
        assert call.args[2] == shard
