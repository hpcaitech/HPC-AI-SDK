# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Required, TypedDict

from .model_id import ModelID
from .forward_backward_input_param import ForwardBackwardInputParam

__all__ = ["TrainingForwardParams"]


class TrainingForwardParams(TypedDict, total=False):
    forward_input: Required[ForwardBackwardInputParam]

    model_id: Required[ModelID]
