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

"""Exercise native vLLM prompt-update and EOS functions without device kernels."""

from __future__ import annotations

import ast
import importlib.util
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
from transformers import GenerationConfig, PreTrainedTokenizerFast


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
            "_apply_matches",
            "apply_token_matches",
        },
        namespace,
    )

    class Base:
        @classmethod
        def __class_getitem__(cls, item):
            return cls

    namespace.update(BaseMultiModalProcessor=Base, KimiK3ProcessingInfo=object, ImageProcessorItems=object)
    _definitions(root / "models/kimi_k3/common/mm_preprocess.py", {"KimiK3MultiModalProcessor"}, namespace)
    _definitions(
        Path(__file__).resolve().parents[3] / "vllm_ascend/models/kimi_k3.py",
        {
            "AscendKimiK3MultiModalProcessor",
        },
        namespace,
    )
    return root, namespace


def _tokenizer():
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "image": 1, "28x14": 2, "14x14": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<|im_end|>",
        additional_special_tokens=[
            "<|kimi_image_placeholder|>",
            "<|media_begin|>",
            "<|media_content|>",
            "<|media_pad|>",
            "<|media_end|>",
        ],
    )


@pytest.mark.parametrize("prompt_format", ["native", "hf"])
@pytest.mark.parametrize("count", [1, 2])
def test_native_prompt_updates_accept_raw_or_checkpoint_media_blocks(native, prompt_format, count):
    _, namespace = native
    tokenizer = _tokenizer()
    images = [Image.new("RGB", size) for size in [(28, 14), (14, 14)][:count]]

    def make_image_prompt(width, height):
        return f"<|media_begin|>image {width}x{height}<|media_content|><|media_pad|><|media_end|>"

    items = SimpleNamespace(
        get_items=lambda *args: SimpleNamespace(
            get=lambda index: images[index],
            get_image_size=lambda index: images[index].size,
        )
    )
    processor = namespace["AscendKimiK3MultiModalProcessor"]()
    processor.info = SimpleNamespace(
        media_token_id=tokenizer.convert_tokens_to_ids("<|media_pad|>"),
        media_token="<|media_pad|>",
        get_hf_config=lambda: SimpleNamespace(image_placeholder="<|kimi_image_placeholder|>"),
        get_tokenizer=lambda: tokenizer,
        image_processor=SimpleNamespace(make_image_prompt=make_image_prompt),
        media_tokens_calculator=lambda media: media["image"].width // 14 * (media["image"].height // 14),
    )
    updates = processor._get_prompt_updates(items, {}, None)
    prompt = "".join(
        "<|kimi_image_placeholder|>" if prompt_format == "native" else make_image_prompt(*image.size)
        for image in images
    )
    original = tokenizer.encode(prompt, add_special_tokens=False)
    expected = [token for index in range(count) for token in updates[0].resolve(index).content.full]
    result, matches = namespace["apply_token_matches"](
        original,
        {
            "image": [[update.resolve(index) for update in updates] for index in range(count)],
        },
    )
    assert matches == {"image": [0 if prompt_format == "native" else 1] * count}
    assert result == expected
    assert result.count(processor.info.media_token_id) == (2 if count == 1 else 3)
    assert original == tokenizer.encode(prompt, add_special_tokens=False)


def test_text_only_prompt_does_not_require_image_items(native):
    _, namespace = native

    def no_images(*args):
        raise AssertionError("Text-only prompts must not query image items")

    processor = namespace["AscendKimiK3MultiModalProcessor"]()
    processor.info = SimpleNamespace(
        media_token_id=1,
        media_token="<|media_pad|>",
        get_hf_config=lambda: SimpleNamespace(image_placeholder="<|kimi_image_placeholder|>"),
        get_tokenizer=_tokenizer,
    )
    updates = processor._get_prompt_updates(SimpleNamespace(get_items=no_images), {}, None)
    assert len(updates) == 2
    result, matches = namespace["apply_token_matches"]([7, 8], {"image": []})
    assert result == [7, 8] and matches == {"image": []}


@pytest.mark.parametrize("ignore_eos", [False, True])
def test_native_renderer_and_generation_config_supply_eos_without_model_hook(native, ignore_eos):
    root, namespace = native
    methods = {}
    for file, owner, name in [
        (root / "renderers/base.py", "BaseRenderer", "get_eos_token_id"),
        (root / "sampling_params.py", "SamplingParams", "update_from_generation_config"),
    ]:
        tree = ast.parse(file.read_text())
        cls = next(node for node in tree.body if getattr(node, "name", None) == owner)
        method = next(node for node in cls.body if getattr(node, "name", None) == name)
        module = ast.Module(
            body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(file), "exec"), methods)
    eos = methods["get_eos_token_id"](SimpleNamespace(tokenizer=_tokenizer()))
    params = SimpleNamespace(ignore_eos=ignore_eos, stop_token_ids=[100], _all_stop_token_ids=set())
    config = GenerationConfig(eos_token_id=[eos, 101]).to_diff_dict()
    methods["update_from_generation_config"](params, config, eos)
    assert params._all_stop_token_ids == {eos, 101}
    assert set(params.stop_token_ids) == ({100} if ignore_eos else {100, 101})
    assert getattr(params, "_eos_token_id", None) == (None if ignore_eos else eos)
