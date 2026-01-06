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

# Path format constants
_URI_SCHEME = "hpcai://"
_ARCHIVE_EXT = ""


class Checkpoint(BaseModel):
    checkpoint_id: str
    """The checkpoint ID"""

    checkpoint_type: CheckpointType
    """The type of checkpoint (training or sampler)"""

    time: datetime
    """The time when the checkpoint was created"""

    checkpoint_path: str
    """The checkpoint path (hpcai://{run_id}/{type}/{ckpt_id})"""


class ParsedCheckpointPath(BaseModel):
    """Parsed checkpoint path.

    Format: hpcai://{training_run_id}/{checkpoint_type}/{checkpoint_id}"""

    checkpoint_path: str
    """The checkpoint path"""

    training_run_id: str
    """The training run ID"""

    checkpoint_type: CheckpointType
    """The type of checkpoint (training or sampler)"""

    checkpoint_id: str
    """The checkpoint ID"""

    @classmethod
    def parse(cls, path: str) -> "ParsedCheckpointPath":
        """Parse a checkpoint path into its components.

        Args:
            path: Checkpoint path like "hpcai://run_xxx/sampler/step_1000"

        Returns:
            ParsedCheckpointPath instance

        Raises:
            ValueError: If path format is invalid
        """
        path_without_scheme = path.removeprefix(_URI_SCHEME)
        path_without_ext = path_without_scheme.removesuffix(_ARCHIVE_EXT)
        parts = path_without_ext.split("/")

        if len(parts) < 3:
            raise ValueError(f"Invalid checkpoint path: {path}")

        checkpoint_type = parts[1]
        if checkpoint_type not in ("training", "sampler"):
            raise ValueError(f"Invalid checkpoint type: {checkpoint_type}")

        return cls(
            checkpoint_path=path,
            training_run_id=parts[0],
            checkpoint_type=checkpoint_type,
            checkpoint_id="/".join(parts[2:]),
        )

    # Backward-compat aliases (older SDK versions exposed these helpers)
    @classmethod
    def from_checkpoint_path(cls, checkpoint_path: str) -> "ParsedCheckpointPath":
        """Alias for parse()."""
        return cls.parse(checkpoint_path)

    @classmethod
    def from_hpcai_path(cls, path: str) -> "ParsedCheckpointPath":
        """Alias for parse()."""
        return cls.parse(path)

    def to_uri(self) -> str:
        """Convert to URI string.

        Returns:
            URI in format: hpcai://{run_id}/{type}/{ckpt_id}"""
        return f"{_URI_SCHEME}{self.training_run_id}/{self.checkpoint_type}/{self.checkpoint_id}{_ARCHIVE_EXT}"

    def to_path(self) -> str:
        """Convert to path string (without URI scheme).

        Returns:
            Path in format: {run_id}/{type}/{ckpt_id}"""
        return f"{self.training_run_id}/{self.checkpoint_type}/{self.checkpoint_id}{_ARCHIVE_EXT}"
