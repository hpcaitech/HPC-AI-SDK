import logging

import hpcai
import torch

logger = logging.getLogger(__name__)


def compute_mean_nll(
    logprobs_list: list[hpcai.TensorData], weights_list: list[hpcai.TensorData]
) -> float:
    """Compute weighted mean negative log likelihood."""
    total_weighted_logprobs = 0.0
    total_weights = 0.0

    for logprobs, weights in zip(logprobs_list, weights_list, strict=True):
        logprobs_torch = logprobs.to_torch()
        weights_torch = weights.to_torch()
        total_weighted_logprobs += logprobs_torch.dot(weights_torch)
        total_weights += weights_torch.sum()

    if total_weights == 0:
        logger.warning("No valid weights found for NLL computation")
        return float("nan")

    return float(-total_weighted_logprobs / total_weights)


def datum_from_tokens_weights(
    tokens: torch.Tensor,
    weights: torch.Tensor,
    max_length: int | None = None,
) -> hpcai.Datum:
    if max_length is not None:
        tokens = tokens[:max_length]
    weights = weights[:max_length]

    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return hpcai.Datum(
        model_input=hpcai.ModelInput.from_ints(tokens=input_tokens.tolist()),
        loss_fn_inputs={
            "weights": hpcai.TensorData(
                data=weights.tolist(),
                dtype="float32",
                shape=list(weights.shape),
            ),
            "target_tokens": hpcai.TensorData(
                data=[int(x) for x in target_tokens.tolist()],
                dtype="int64",
                shape=list(target_tokens.shape),
            ),
        },
    )
