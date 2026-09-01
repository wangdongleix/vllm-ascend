#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

import os

_GLOBAL_PATCH_APPLIED = False


def _apply_vllm_source_version_override() -> None:
    """Make source-tree vLLM report the reviewed version without editing it.

    The container still has a 0.23 editable distribution whose generated
    ``vllm._version`` module is visible even when the 0.26 source tree is first
    on PYTHONPATH.  ``VLLM_VERSION`` is already the vLLM-Ascend compatibility
    override; apply it to vLLM's public version module as well so verl and
    vLLM's cache/startup fingerprints do not branch on stale metadata.
    """
    source_version = os.getenv("VLLM_VERSION")
    if not source_version:
        return

    from packaging.version import Version

    parsed = Version(source_version)
    import vllm
    import vllm.version as vllm_version_module

    version_tuple = tuple(parsed.release)
    vllm.__version__ = source_version
    vllm.__version_tuple__ = version_tuple
    vllm_version_module.__version__ = source_version
    vllm_version_module.__version_tuple__ = version_tuple


def _ensure_global_patch():
    """Apply process-wide vLLM patches before engine-core initialization.

    vLLM loads general plugins in engine-core subprocesses. E2E test
    conftest hooks do not run there, so global patches that affect scheduler
    and engine code must also be applied through these plugin entry points.
    """
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED:
        return

    _apply_vllm_source_version_override()

    from vllm_ascend.utils import adapt_patch

    adapt_patch(is_global_patch=True)
    # The routed-experts protocol is allocated in both EngineCore and worker
    # processes.  Install Kimi's full-R3 wire width from the general plugin,
    # before either side constructs its buffer.
    from vllm_ascend.patch.kimi_full_r3_schema import (
        install_kimi_full_r3_schema_patch,
    )

    install_kimi_full_r3_schema_patch()
    _GLOBAL_PATCH_APPLIED = True


def register():
    """Register the NPU platform."""

    _apply_vllm_source_version_override()

    return "vllm_ascend.platform.NPUPlatform"


def register_connector():
    _ensure_global_patch()

    from vllm_ascend.distributed.kv_transfer import register_connector
    from vllm_ascend.distributed.weight_transfer import register_engine

    register_connector()
    register_engine()


def register_model_loader():
    _ensure_global_patch()

    from .model_loader.netloader import register_netloader
    from .model_loader.rfork import register_rforkloader

    register_netloader()
    register_rforkloader()


def register_service_profiling():
    _ensure_global_patch()

    from .profiling_config import generate_service_profiling_config

    generate_service_profiling_config()


def register_model():
    _ensure_global_patch()

    from vllm_ascend.transformers_utils.configs.kimi_k3 import register_kimi_k3_config

    register_kimi_k3_config()

    from vllm_ascend.patch.hunyuan_vl_processor_compat import (
        install_hunyuan_vl_processor_compat,
    )

    from .models import register_model

    install_hunyuan_vl_processor_compat()

    register_model()


import vllm_ascend.logger  # noqa: E402, F401
