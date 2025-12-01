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
from typing_extensions import Literal, Required, TypedDict

__all__ = ["EncodedTextChunkParam"]


class EncodedTextChunkParam(TypedDict, total=False):
    tokens: Required[Iterable[int]]
    """Array of token IDs"""

    type: Required[Literal["encoded_text"]]
