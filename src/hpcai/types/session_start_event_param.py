# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing import Union
from datetime import datetime
from typing_extensions import Required, Annotated, TypedDict

from .._utils import PropertyInfo
from .severity import Severity
from .event_type import EventType

__all__ = ["SessionStartEventParam"]


class SessionStartEventParam(TypedDict, total=False):
    event: Required[EventType]
    """Telemetry event type"""

    event_id: Required[str]

    event_session_index: Required[int]

    severity: Required[Severity]
    """Log severity level"""

    timestamp: Required[Annotated[Union[str, datetime], PropertyInfo(format="iso8601")]]
