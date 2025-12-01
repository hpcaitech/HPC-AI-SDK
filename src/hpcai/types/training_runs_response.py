# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

from .._models import BaseModel
from .cursor import Cursor
from .training_run import TrainingRun

__all__ = ["TrainingRunsResponse"]


class TrainingRunsResponse(BaseModel):
    training_runs: list[TrainingRun]
    """List of training runs"""

    cursor: Cursor
    """Pagination cursor information"""
