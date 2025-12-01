# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Iterable
from typing_extensions import Required, TypedDict

from .telemetry_event_param import TelemetryEventParam

__all__ = ["TelemetrySendParams"]


class TelemetrySendParams(TypedDict, total=False):
    events: Required[Iterable[TelemetryEventParam]]

    platform: Required[str]
    """Host platform name"""

    sdk_version: Required[str]
    """SDK version string"""

    session_id: Required[str]
