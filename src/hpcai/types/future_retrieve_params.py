# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Required, TypedDict

from .model_id import ModelID
from .request_id import RequestID

__all__ = ["FutureRetrieveParams"]


class FutureRetrieveParams(TypedDict, total=False):
    request_id: Required[RequestID]

    model_id: ModelID
