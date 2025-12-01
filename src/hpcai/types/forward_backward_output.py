# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Dict, List

from .._models import BaseModel
from .loss_fn_output import LossFnOutput

__all__ = ["ForwardBackwardOutput"]


class ForwardBackwardOutput(BaseModel):
    loss_fn_output_type: str
    """The type of the ForwardBackward output. Can be one of [...] TODO"""

    loss_fn_outputs: List[LossFnOutput]
    """Dictionary mapping field names to tensor data"""

    metrics: Dict[str, float]
    """Training metrics as key-value pairs"""
