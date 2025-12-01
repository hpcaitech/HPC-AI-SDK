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

from .model_input_chunk_param import ModelInputChunkParam

__all__ = ["ModelInputParam"]


class ModelInputParam(TypedDict, total=False):
    chunks: Required[Iterable[ModelInputChunkParam]]
    """Sequence of input chunks (formerly TokenSequence)"""
