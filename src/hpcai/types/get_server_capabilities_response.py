# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import List, Optional

from .._models import BaseModel

__all__ = ["GetServerCapabilitiesResponse", "SupportedModel"]


class SupportedModel(BaseModel):
    model_name: Optional[str] = None


class GetServerCapabilitiesResponse(BaseModel):
    supported_models: List[SupportedModel]
