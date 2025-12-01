# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from typing_extensions import TypeAlias

from .encoded_text_chunk_param import EncodedTextChunkParam
from .image_asset_pointer_chunk_param import ImageAssetPointerChunkParam

__all__ = ["ModelInputChunkParam"]

ModelInputChunkParam: TypeAlias = Union[EncodedTextChunkParam, ImageAssetPointerChunkParam]
