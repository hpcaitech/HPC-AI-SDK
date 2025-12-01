# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from .._models import BaseModel

__all__ = ["LoadWeightsResponse"]


class LoadWeightsResponse(BaseModel):
    path: Optional[str] = None
    """A hpcai URI for model weights at a specific step"""

    type: Optional[Literal["load_weights"]] = None
