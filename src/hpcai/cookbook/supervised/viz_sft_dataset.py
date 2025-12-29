# Copyright 2025 Thinking Machines Lab
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM
"""
Script to visualize supervised datasets in the terminal.
"""

import chz
from hpcai.cookbook import model_info
from hpcai.cookbook.supervised.types import (
    ChatDatasetBuilderCommonConfig,
    SupervisedDatasetBuilder,
)
from hpcai.cookbook.tokenizer_utils import get_tokenizer
from hpcai.cookbook.utils.format_colorized import format_colorized
from hpcai.cookbook.utils.misc_utils import lookup_func
from hpcai.cookbook.renderers import TrainOnWhat


@chz.chz
class Config:
    model_name: str = "meta-llama/Llama-3.1-8B"  # just for tokenizer
    dataset_path: str = "Tulu3Builder"
    renderer_name: str | None = None
    max_length: int | None = None
    train_on_what: TrainOnWhat | None = None


def run(cfg: Config):
    n_examples_total = 100
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.renderer_name or model_info.get_recommended_renderer_name(cfg.model_name),
        max_length=cfg.max_length,
        batch_size=n_examples_total,
        train_on_what=cfg.train_on_what,
    )
    dataset_builder = lookup_func(
        cfg.dataset_path, default_module="hpcai.cookbook.recipes.chat_sl.chat_datasets"
    )(common_config=common_config)
    assert isinstance(dataset_builder, SupervisedDatasetBuilder)
    tokenizer = get_tokenizer(cfg.model_name)
    train_dataset, _ = dataset_builder()
    batch = train_dataset.get_batch(0)

    for datum in batch:
        int_tokens = list(datum.model_input.to_ints()) + [
            datum.loss_fn_inputs["target_tokens"].tolist()[-1]
        ]
        weights = [0.0] + datum.loss_fn_inputs["weights"].tolist()
        print(format_colorized(int_tokens, weights, tokenizer))
        input("press enter")


if __name__ == "__main__":
    chz.nested_entrypoint(run)
