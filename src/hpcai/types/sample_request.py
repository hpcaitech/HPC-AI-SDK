# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

# File generated from our OpenAPI spec by Stainless. See CONTRIBUTING.md for details.

from typing import Optional
from typing_extensions import Literal

from .._compat import PYDANTIC_V2, ConfigDict
from .._models import StrictBase
from .model_input import ModelInput
from .sampling_params import SamplingParams

__all__ = ["SampleRequest"]


class SampleRequest(StrictBase):
    num_samples: int = 1
    """Number of samples to generate"""

    prompt: ModelInput

    sampling_params: SamplingParams

    base_model: Optional[str] = None
    """Optional base model name to sample from.

    Is inferred from model_path, if provided. If sampling against a base model, this
    is required.
    """

    model_path: Optional[str] = None
    """Optional hpcai:// path to your model weights or LoRA weights.

    If not provided, samples against the base model.
    """

    prompt_logprobs: Optional[bool] = None
    """If set to `true`, computes and returns logprobs on the prompt tokens.

    Defaults to false.
    """

    type: Optional[Literal["sample"]] = None

    if PYDANTIC_V2:
        # allow fields with a `model_` prefix
        model_config = ConfigDict(protected_namespaces=tuple())
