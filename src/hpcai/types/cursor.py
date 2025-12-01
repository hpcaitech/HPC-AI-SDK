# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

from .._models import BaseModel

__all__ = ["Cursor"]


class Cursor(BaseModel):
    offset: int
    """The offset used for pagination"""

    limit: int
    """The maximum number of items requested"""

    total_count: int
    """The total number of items available"""
