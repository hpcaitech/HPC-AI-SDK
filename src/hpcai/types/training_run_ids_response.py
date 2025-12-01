# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

from .._models import BaseModel

__all__ = ["TrainingRunIdsResponse"]


class TrainingRunIdsResponse(BaseModel):
    training_run_ids: list[str]
    """List of training run IDs"""

    has_more: bool
    """Whether there are more results available for pagination"""
