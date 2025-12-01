# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .telemetry import Telemetry


@runtime_checkable
class TelemetryProvider(Protocol):
    def get_telemetry(self) -> Telemetry | None: ...
