# Copyright 2026 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2026 HPC-AI.COM

from typing_extensions import Literal
from typing_extensions import TypedDict

__all__ = ["ImageChunkParam"]


class ImageChunkParam(TypedDict, total=False):
    data: bytes
    """Image data as bytes"""

    format: Literal["png", "jpeg"]
    """Image format"""

    expected_tokens: int | None = None
    """Expected number of tokens this image represents.
    This is only advisory: the hpcai backend will compute the number of tokens
    from the image, and we can fail requests quickly if the tokens does not
    match expected_tokens."""

    type: Literal["image"] = "image"