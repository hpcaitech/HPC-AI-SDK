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

__all__ = ["WeightLoadParams"]


class WeightLoadParams(TypedDict, total=False):
    model_id: Required[ModelID]

    path: Required[str]
    """A hpcai URI for model weights at a specific step"""

    type: Literal["load_weights"] = "load_weights"
