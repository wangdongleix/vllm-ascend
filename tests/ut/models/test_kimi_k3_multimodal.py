# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from PIL import Image
from transformers import BatchFeature
from vllm.config.multimodal import ImageDummyOptions
from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFlatField,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import (
    MultiModalDataItems,
    VisionChunkProcessorItems,
)
from vllm.multimodal.processing import BaseMultiModalProcessor

import vllm_ascend.models.kimi_k3 as kimi_k3_module
from vllm_ascend.models.kimi_k3 import (
    AscendKimiK3ForConditionalGeneration,
    KimiK3DummyInputsBuilder,
    KimiK3MultiModalProcessor,
    KimiK3ProcessingInfo,
    navit_resize_image,
)
from vllm_ascend.transformers_utils.processors.kimi_k3 import KimiK3Processor

IMAGE_PLACEHOLDER = (
    "<|media_begin|>image<|media_content|>"
    "<|media_pad|><|media_end|>"
)
MEDIA_TOKEN_ID = 163605


class _EmptyLoadedNamesLoader:
    """Model the vLLM path that loads a nested Linear but omits its name."""

    def __init__(self, *args, **kwargs):
        del args, kwargs

    def load_weights(self, weights, mapper):
        del mapper
        list(weights)
        return set()


def _projector_load_test_model(*, use_rot_proj: bool):
    model = object.__new__(AscendKimiK3ForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(
        use_rot_proj=False,
        vision_config=SimpleNamespace(use_rot_proj=use_rot_proj),
    )
    model.mm_projector = torch.nn.Module()
    model.mm_projector.rot_proj = torch.nn.Linear(2, 2, bias=False)
    return model


def test_kimi_k3_load_keeps_configured_rot_proj_when_loader_omits_name(
    monkeypatch,
):
    monkeypatch.setattr(
        kimi_k3_module,
        "AutoWeightsLoader",
        _EmptyLoadedNamesLoader,
    )
    model = _projector_load_test_model(use_rot_proj=True)

    assert model.load_weights([]) == set()
    assert isinstance(model.mm_projector.rot_proj, torch.nn.Linear)


def test_kimi_k3_load_drops_unconfigured_absent_rot_proj(monkeypatch):
    monkeypatch.setattr(
        kimi_k3_module,
        "AutoWeightsLoader",
        _EmptyLoadedNamesLoader,
    )
    model = _projector_load_test_model(use_rot_proj=False)

    assert model.load_weights([]) == set()
    assert not hasattr(model.mm_projector, "rot_proj")


def test_kimi_k3_processor_consumes_vision_chunks_without_expanding_prompt():
    images = [object(), object()]
    vision_chunks = [
        {"type": "image", "image": images[0]},
        {"type": "image", "image": images[1]},
    ]
    image_processor = SimpleNamespace(
        preprocess=MagicMock(
            return_value={
                "pixel_values": torch.ones(2, 3),
                "grid_thws": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            }
        )
    )
    tokenizer = MagicMock(
        return_value={
            "input_ids": [[101, 102, 103]],
            "attention_mask": [[1, 1, 1]],
        }
    )
    processor = KimiK3Processor(image_processor, tokenizer)

    outputs = processor(
        text=f"before {IMAGE_PLACEHOLDER} after",
        vision_chunks=vision_chunks,
        return_tensors=None,
    )

    image_processor.preprocess.assert_called_once()
    assert image_processor.preprocess.call_args.args[0] is vision_chunks
    assert image_processor.preprocess.call_args.kwargs == {
        "return_tensors": None,
    }
    tokenizer.assert_called_once_with(
        [f"before {IMAGE_PLACEHOLDER} after"],
    )
    assert outputs["input_ids"] == [[101, 102, 103]]
    assert outputs["attention_mask"] == [[1, 1, 1]]


def test_kimi_k3_multimodal_fields_use_vision_chunk_and_grid_slices():
    # The generic MM-only helper handles image/video/audio, not vision_chunk.
    # This override selects vLLM's text+MM path and must not be deduplicated.
    assert (
        KimiK3MultiModalProcessor._call_hf_processor
        is not BaseMultiModalProcessor._call_hf_processor
    )
    processor = object.__new__(KimiK3MultiModalProcessor)
    grid_thws = torch.tensor([[1, 2, 3], [2, 3, 4]])

    assert (
        processor._hf_processor_applies_updates(
            prompt_text=IMAGE_PLACEHOLDER,
            mm_items=MagicMock(),
            hf_processor_mm_kwargs={},
            tokenization_kwargs={},
        )
        is False
    )

    fields = processor._get_mm_fields_config(
        BatchFeature({"grid_thws": grid_thws}),
        {},
    )

    pixel_values = fields["pixel_values"]
    assert pixel_values.modality == "vision_chunk"
    assert isinstance(pixel_values.field, MultiModalFlatField)
    assert [(int(item[0].start), int(item[0].stop)) for item in pixel_values.field.slices] == [(0, 6), (6, 30)]

    grid = fields["grid_thws"]
    assert grid.modality == "vision_chunk"
    assert isinstance(grid.field, MultiModalBatchedField)
    assert grid.field.keep_on_cpu is True


def test_kimi_k3_prompt_update_expands_original_image_size_and_media_pads():
    image = Image.new("RGB", (640, 480))
    media_tokens_calculator = MagicMock(return_value=3)
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=False: {
        "<|media_begin|>image<|media_content|>": [10],
        "<|media_begin|>image 640x480<|media_content|>": [20],
        "<|media_end|>": [11],
    }[text]
    processor = object.__new__(KimiK3MultiModalProcessor)
    processor.info = SimpleNamespace(
        media_token_id=MEDIA_TOKEN_ID,
        media_tokens_calculator=media_tokens_calculator,
        get_tokenizer=lambda: tokenizer,
    )
    mm_items = MultiModalDataItems(
        {
            "vision_chunk": VisionChunkProcessorItems(
                [{"type": "image", "image": image}]
            )
        },
    )

    updates = processor._get_prompt_updates(
        mm_items,
        {},
        MultiModalKwargsItems({}),
    )

    assert len(updates) == 1
    update = updates[0]
    assert update.modality == "vision_chunk"
    assert update.target == [10, MEDIA_TOKEN_ID, 11]
    assert callable(update.replacement)

    details = update.replacement(0)
    assert details.full == [20, MEDIA_TOKEN_ID, MEDIA_TOKEN_ID, MEDIA_TOKEN_ID, 11]
    media_tokens_calculator.assert_called_once()
    media = media_tokens_calculator.call_args.args[0]
    assert media["type"] == "image"
    assert media["image"] is image

    assert details.is_embed is not None
    assert details.is_embed(tokenizer, details.full).tolist() == [
        False,
        True,
        True,
        True,
        False,
    ]


def test_kimi_k3_dummy_builder_profiles_true_maximum_image_shape():
    size = KimiK3ProcessingInfo.get_max_image_size(
        patch_size=14,
        merge_kernel_size=2,
        in_patch_limit=65536,
        patch_limit_on_one_side=512,
        fixed_output_tokens=None,
    )
    assert size == (1861, 7041)
    assert (
        navit_resize_image(
            size.width,
            size.height,
            patch_size=14,
            merge_kernel_size=2,
            in_patch_limit=65536,
            patch_limit_on_one_side=512,
            fixed_output_tokens=None,
        )["num_tokens"]
        == 16817
    )

    info = SimpleNamespace(
        image_processor=SimpleNamespace(
            media_proc_cfg={
                "patch_size": 14,
                "merge_kernel_size": 2,
                "in_patch_limit": 65536,
                "patch_limit_on_one_side": 512,
                "fixed_output_tokens": None,
            }
        ),
        get_max_image_size=KimiK3ProcessingInfo.get_max_image_size,
    )
    builder = KimiK3DummyInputsBuilder(info)
    builder._get_dummy_images = MagicMock(return_value=["image-0", "image-1"])
    options = ImageDummyOptions(count=2, width=640, height=480)

    assert KimiK3ProcessingInfo.get_supported_mm_limits(object()) == {
        "vision_chunk": None
    }
    assert builder.get_dummy_text({"vision_chunk": 2}) == IMAGE_PLACEHOLDER * 2
    assert builder.get_dummy_mm_data(
        seq_len=4096,
        mm_counts={"vision_chunk": 2},
        mm_options={"vision_chunk": options},
    ) == {
        "vision_chunk": [
            {"type": "image", "image": "image-0"},
            {"type": "image", "image": "image-1"},
        ]
    }
    builder._get_dummy_images.assert_called_once_with(
        height=7041,
        width=1861,
        num_images=2,
        overrides=options,
    )
