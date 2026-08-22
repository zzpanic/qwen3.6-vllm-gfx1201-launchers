# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# PORTED FROM vLLM upstream PR #52816, merged as b389ac29465b33f9e9c534df221ea3c129e9793f
# Base: vLLM 0.26.0 (docker.io/stilldeadcode/vllm-radiance:0.5.8)
# Ported 2026-08-21 for qwen3.8-27b-vllm DFlash2 evaluation. See patches/dflash2/README.md
# DEVIATIONS FROM MERGED UPSTREAM IN THIS FILE: none
# HAND-ADAPTED HUNKS (0.26.0 drift): the PR's hunk context included a preceding
#   "elif ... ExtractHiddenStatesSpeculator" branch that does not exist in 0.26.0,
#   which offset the diff and caused a naive fuzzy-patch to misfile the new
#   DFlash2DraftModel dispatch check under the "dspark" elif branch instead of the
#   "dflash" branch. Hand-corrected: the
#   `if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:`
#   check (and its DFlash2Speculator import/return) now sits inside
#   `if speculative_config.method == "dflash":`, before the DFlashSpeculator
#   fallback import — matching the PR's actual intent (dflash2 is a dflash-method
#   draft that dispatches to a different speculator class based on architecture).
import torch

from vllm.config import VllmConfig


def init_speculator(vllm_config: VllmConfig, device: torch.device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if speculative_config.method == "dflash":
        if "DFlash2DraftModel" in speculative_config.draft_model_config.architectures:
            from vllm.v1.worker.gpu.spec_decode.dflash2.speculator import (
                DFlash2Speculator,
            )

            return DFlash2Speculator(vllm_config, device)
        from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
            DFlashSpeculator,
        )

        return DFlashSpeculator(vllm_config, device)
    elif speculative_config.method == "dspark":
        from vllm.v1.worker.gpu.spec_decode.dspark.speculator import (
            DSparkSpeculator,
        )

        return DSparkSpeculator(vllm_config, device)
    elif speculative_config.use_gemma4_mtp():
        from vllm.v1.worker.gpu.spec_decode.gemma4.speculator import (
            Gemma4Speculator,
        )

        return Gemma4Speculator(vllm_config, device)
    elif speculative_config.method == "mtp":
        from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator

        return MTPSpeculator(vllm_config, device)
    elif speculative_config.use_eagle():
        from vllm.v1.worker.gpu.spec_decode.eagle.speculator import (
            EagleSpeculator,
        )

        return EagleSpeculator(vllm_config, device)
    else:
        raise NotImplementedError(f"{speculative_config.method} is not supported yet.")
