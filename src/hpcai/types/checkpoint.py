# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

from datetime import datetime
from typing import Literal

from .._models import BaseModel

__all__ = ["Checkpoint", "CheckpointType", "ParsedCheckpointPath"]

CheckpointType = Literal["training", "sampler"]


class Checkpoint(BaseModel):
    checkpoint_id: str
    """The checkpoint ID"""

    checkpoint_type: CheckpointType
    """The type of checkpoint (training or sampler)"""

    time: datetime
    """The time when the checkpoint was created"""

    checkpoint_path: str
    """The checkpoint path (must use the hpcai:// protocol)"""


class ParsedCheckpointPath(BaseModel):
    """Parsed checkpoint path for the hpcai:// protocol."""

    checkpoint_path: str
    """The checkpoint path (hpcai:// protocol)"""

    training_run_id: str
    """The training run ID"""

    checkpoint_type: CheckpointType
    """The type of checkpoint (training or sampler)"""

    checkpoint_id: str
    """The checkpoint ID"""

    @classmethod
    def from_checkpoint_path(cls, checkpoint_path: str) -> "ParsedCheckpointPath":
        """Parse a checkpoint path into its components.

        Args:
            checkpoint_path: Path with the hpcai:// protocol

        Returns:
            ParsedCheckpointPath instance

        Raises:
            ValueError: If path format is invalid
        """

        if not checkpoint_path.startswith("hpcai://"):
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}. Must start with 'hpcai://'.")

        parts = checkpoint_path[8:].split("/")
        if len(parts) != 3:
            raise ValueError(f"Invalid checkpoint path format: {checkpoint_path}")
        if parts[1] not in ["weights", "sampler_weights"]:
            raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")
        checkpoint_type = "training" if parts[1] == "weights" else "sampler"
        return cls(
            checkpoint_path=checkpoint_path,
            training_run_id=parts[0],
            checkpoint_type=checkpoint_type,
            checkpoint_id="/".join(parts[2:]),
        )

    @classmethod
    def from_hpcai_path(cls, path: str) -> "ParsedCheckpointPath":
        """Alias for from_checkpoint_path using the hpcai:// protocol."""

        return cls.from_checkpoint_path(path)
