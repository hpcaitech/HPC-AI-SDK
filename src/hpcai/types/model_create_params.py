# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from __future__ import annotations

from typing_extensions import Literal, Required, TypedDict

from .lora_config_param import LoraConfigParam

__all__ = ["ModelCreateParams"]


class ModelCreateParams(TypedDict, total=False):
    base_model: Required[str]

    lora_config: LoraConfigParam

    type: Literal["create_model"] = "create_model"
