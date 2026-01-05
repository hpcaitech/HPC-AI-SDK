# Copyright 2025 Thinking Machines Lab
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM
import logging
from typing import Callable

import hpcai

# Set up logger
logger = logging.getLogger(__name__)


class TrainingClientEvaluator:
    """
    An evaluator that takes in a TrainingClient
    """

    async def __call__(self, training_client: hpcai.TrainingClient) -> dict[str, float]:
        raise NotImplementedError


class SamplingClientEvaluator:
    """
    An evaluator that takes in a TokenCompleter
    """

    async def __call__(self, sampling_client: hpcai.SamplingClient) -> dict[str, float]:
        raise NotImplementedError


EvaluatorBuilder = Callable[[], TrainingClientEvaluator | SamplingClientEvaluator]
SamplingClientEvaluatorBuilder = Callable[[], SamplingClientEvaluator]
Evaluator = TrainingClientEvaluator | SamplingClientEvaluator
