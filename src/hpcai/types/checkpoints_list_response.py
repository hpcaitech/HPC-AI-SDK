# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

from .._models import BaseModel
from .checkpoint import Checkpoint

__all__ = ["CheckpointsListResponse"]


class CheckpointsListResponse(BaseModel):
    checkpoints: list[Checkpoint]
    """List of available model checkpoints for the model"""
