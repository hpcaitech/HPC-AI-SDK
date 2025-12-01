# Copyright 2025 Thinking Machines Lab
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

"""
Docstring for Colossal-Tinker-backend.colossal_tinker.tinker-local-dev.src.hpcai.cookbook.data
"""

import torch
from hpcai.types import Datum, ModelInput, TensorData
from hpcai.cookbook.renderers import Message, Renderer, TrainOnWhat
from hpcai.cookbook.common import datum_from_tokens_weights


def datum_from_tokens_weights(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    max_length: int | None = None,
) -> Datum:
    if max_length is not None:
        tokens = tokens[:max_length]
    weights = weights[:max_length]

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return Datum(
        model_input=ModelInput.from_ints(tokens=input_tokens.tolist()),
        loss_fn_inputs={
            "weights": TensorData(
                data=weights.tolist(),
                dtype="float32",
                shape=list(weights.shape),
            ),
            "target_tokens": TensorData(
                data=[int(x) for x in target_tokens.tolist()],
                dtype="int64",
                shape=list(target_tokens.shape),
            ),
        },
    )


def conversation_to_datum(
    conversation: list[Message],
    renderer: Renderer,
    max_length: int | None,
    train_on_what: TrainOnWhat = TrainOnWhat.ALL_ASSISTANT_MESSAGES,
) -> Datum:
    """Common function to process a list of messages into a Datum."""
    tokens, weights = renderer.build_supervised_example(conversation, train_on_what=train_on_what)
    return datum_from_tokens_weights(tokens, weights, max_length)
