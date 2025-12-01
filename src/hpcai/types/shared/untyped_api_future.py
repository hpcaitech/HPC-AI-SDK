# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional

from ..._compat import PYDANTIC_V2, ConfigDict
from ..._models import BaseModel
from ..model_id import ModelID
from ..request_id import RequestID

__all__ = ["UntypedAPIFuture"]


class UntypedAPIFuture(BaseModel):
    request_id: RequestID

    model_id: Optional[ModelID] = None
