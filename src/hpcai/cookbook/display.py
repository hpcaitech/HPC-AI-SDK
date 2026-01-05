# Copyright 2025 Thinking Machines Lab
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM
import io

import hpcai
from termcolor import colored

from hpcai.cookbook.rl.types import Trajectory
from hpcai.cookbook.tokenizer_utils import Tokenizer
from hpcai.cookbook.utils.format_colorized import format_colorized


def to_ints(chunk: hpcai.ModelInputChunk, tokenizer: Tokenizer):
    if isinstance(chunk, hpcai.EncodedTextChunk):
        return chunk.tokens
    else:
        (at_token,) = tokenizer.encode("@", add_special_tokens=False)
        return [at_token] * chunk.length


def colorize_example(datum: hpcai.Datum, tokenizer: Tokenizer, key: str = "weights"):
    int_tokens = [
        token for chunk in datum.model_input.chunks for token in to_ints(chunk, tokenizer)
    ] + [datum.loss_fn_inputs["target_tokens"].tolist()[-1]]
    weights = [0.0] + datum.loss_fn_inputs[key].tolist()
    return format_colorized(int_tokens, weights, tokenizer)


def format_trajectory(trajectory: Trajectory, tokenizer: Tokenizer) -> str:
    buf = io.StringIO()

    def colorize(s: str):
        return colored(s, "green", attrs=["bold"])

    def bprint(s: str):
        print(s, file=buf)

    bprint("=" * 60)
    for i, transition in enumerate(trajectory.transitions):
        bprint(f"------ Transition {i} ------")
        bprint(f"{colorize('Observation:')}: {tokenizer.decode(transition.ob.to_ints())}")
        bprint(f"{colorize('Action:')}: {tokenizer.decode(transition.ac.tokens)}")
        bprint(f"{colorize('Reward:')}: {transition.reward}")
        bprint(f"{colorize('Episode done:')}: {transition.episode_done}")
        bprint(f"{colorize('Metrics:')}: {transition.metrics}")
        bprint("-" * 60)
    bprint("=" * 60)
    return buf.getvalue()
