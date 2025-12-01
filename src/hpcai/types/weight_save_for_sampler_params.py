# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .model_id import ModelID

__all__ = ["WeightSaveForSamplerParams"]


class WeightSaveForSamplerParams(TypedDict, total=False):
    model_id: Required[ModelID]

    path: str
    """A file/directory name for the weights"""

    type: Literal["save_weights_for_sampler"] = "save_weights_for_sampler"
