# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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

"""Exercise native vLLM prompt updates without device kernels."""

from __future__ import annotations

import ast
import importlib.util
import math
import os
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast


def _definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *nodes],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)


@pytest.fixture
def native():
    source = os.environ.get("VLLM_SOURCE_ROOT")
    if source:
        root = Path(source) / "vllm"
    else:
        spec = importlib.util.find_spec("vllm")
        if spec is None or spec.origin is None:
            pytest.skip("requires vLLM source or VLLM_SOURCE_ROOT")
        root = Path(spec.origin).parent
    namespace = dict(
        torch=torch,
        math=math,
        ABC=ABC,
        abstractmethod=abstractmethod,
        deque=deque,
        dataclass=dataclass,
        field=field,
        replace=replace,
        Enum=Enum,
        NamedTuple=NamedTuple,
        lru_cache=lru_cache,
        _UpdateQueue=deque,
        _QueueMatch=tuple,
        flatten_2d_lists=lambda values: list(chain.from_iterable(values)),
    )
    _definitions(
        root / "multimodal/processing/processor.py",
        {
            "cached_encode",
            "PromptIndex",
            "PromptUpdateDetails",
            "UpdateMode",
            "PromptUpdate",
            "PromptReplacement",
            "PromptTargetMatch",
            "ResolvedPromptUpdate",
            "_TokenMatch",
            "iter_token_matches",
            "_MatchedUpdate",
            "_target_key",
            "_compile_prompt_update_queues",
            "_find_queue_match",
            "_next_priority",
            "_plan_prompt_updates_with",
            "_plan_prompt_updates",
            "PlaceholderFeaturesInfo",
            "_apply_token_matches_with_placeholders",
        },
        namespace,
    )

    class Base:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

    namespace.update(BaseMultiModalProcessor=Base, KimiK3ProcessingInfo=object, ImageProcessorItems=object)
    _definitions(
        root / "models/kimi_k3/common/mm_preprocess.py", {"KimiK3MultiModalProcessor", "navit_resize_image"}, namespace
    )
    _definitions(
        Path(__file__).resolve().parents[3] / "vllm_ascend/models/kimi_k3.py",
        {
            "AscendKimiK3MultiModalProcessor",
        },
        namespace,
    )
    return root, namespace


def _tokenizer():
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "image": 1, "56x28": 2, "28x28": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        additional_special_tokens=[
            "<|kimi_image_placeholder|>",
            "<|media_begin|>",
            "<|media_content|>",
            "<|media_pad|>",
            "<|media_end|>",
        ],
    )


@pytest.mark.parametrize(
    "count,prompt_format",
    [(0, "native"), (1, "native"), (1, "hf"), (2, "native"), (2, "hf"), (1, "expanded"), (2, "expanded")],
)
def test_native_prompt_updates_accept_raw_or_checkpoint_media_blocks(native, prompt_format, count):
    _, namespace = native
    tokenizer = _tokenizer()
    images = [Image.new("RGB", size) for size in [(56, 28), (28, 28)][:count]]

    def make_image_prompt(width, height, pads=1):
        return f"<|media_begin|>image {width}x{height}<|media_content|>" + "<|media_pad|>" * pads + "<|media_end|>"

    def feature_count(media):
        return namespace["navit_resize_image"](*media["image"].size, 14, 2, 64, 8, None)["num_tokens"]

    def get_items(*args):
        assert images, "Text-only prompts must not query image items"
        return SimpleNamespace(
            get=lambda index: images[index],
            get_image_size=lambda index: images[index].size,
        )

    items = SimpleNamespace(get_items=get_items)
    processor = namespace["AscendKimiK3MultiModalProcessor"]()
    processor.info = SimpleNamespace(
        media_token_id=tokenizer.convert_tokens_to_ids("<|media_pad|>"),
        media_token="<|media_pad|>",
        get_hf_config=lambda: SimpleNamespace(image_placeholder="<|kimi_image_placeholder|>"),
        get_tokenizer=lambda: tokenizer,
        image_processor=SimpleNamespace(make_image_prompt=make_image_prompt),
        media_tokens_calculator=feature_count,
    )
    updates = processor._get_prompt_updates(items, {}, None)
    prompt = "".join(
        "<|kimi_image_placeholder|>"
        if prompt_format == "native"
        else make_image_prompt(*image.size, pads=(2, 1)[index] if prompt_format == "expanded" else 1)
        for index, image in enumerate(images)
    )
    original = tokenizer.encode(prompt + " image", add_special_tokens=False)
    expected = [token for index in range(count) for token in updates[0].resolve(index).content.full] + [1]
    result, matches, placeholders = namespace["_apply_token_matches_with_placeholders"](
        original,
        {
            "image": [[update.resolve(index) for update in updates] for index in range(count)],
        },
    )
    expected_matches = [0 if prompt_format == "native" else 1] * count
    if prompt_format == "expanded":
        expected_matches = [2, 1][:count]
        assert result == original
    assert matches == {"image": expected_matches}
    assert [int(item.is_embed.sum()) for item in placeholders["image"]] == [2, 1][:count]
    assert result == expected
    assert result.count(processor.info.media_token_id) == (0, 2, 3)[count]
    assert original == tokenizer.encode(prompt + " image", add_special_tokens=False)


@pytest.mark.parametrize(
    "kwargs", [{}, {"images_kwargs": {}}, {"return_tensors": "pt", "truncation": False, "padding": False}]
)
def test_native_image_boundary_preserves_framework_kwargs(native, kwargs):
    _, namespace = native
    processor = namespace["AscendKimiK3MultiModalProcessor"]()
    data = {"images": [Image.new("RGB", (56, 28))]}
    result = processor._preprocess_hf_mm_data(data, kwargs)
    assert result[0] is data and result[1] is kwargs


@pytest.mark.parametrize(
    "kwargs",
    [
        {"images_kwargs": {"size": {"width": 28, "height": 28}}},
        {"size": {"width": 28, "height": 28}},
        {"in_patch_limit": 16},
    ],
)
def test_native_image_boundary_rejects_ignored_resize_overrides(native, kwargs):
    _, namespace = native
    processor = namespace["AscendKimiK3MultiModalProcessor"]()
    with pytest.raises(TypeError, match="does not support image preprocessing overrides"):
        processor._preprocess_hf_mm_data({"images": [Image.new("RGB", (56, 28))]}, kwargs)
