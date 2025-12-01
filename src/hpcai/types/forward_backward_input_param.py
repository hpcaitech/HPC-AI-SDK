# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Iterable
from typing_extensions import Required, TypedDict

from .datum_param import DatumParam
from .loss_fn_type import LossFnType

__all__ = ["ForwardBackwardInputParam"]


class ForwardBackwardInputParam(TypedDict, total=False):
    data: Required[Iterable[DatumParam]]
    """Array of input data for the forward/backward pass"""

    loss_fn: Required[LossFnType]
    """Fully qualified function path for the loss function"""
