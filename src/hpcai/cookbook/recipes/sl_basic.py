# Copyright 2025 Thinking Machines Lab
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM
import chz
import sys
from hpcai.cookbook import cli_utils, model_info
from hpcai.cookbook.recipes.chat_sl import chat_datasets
from hpcai.cookbook.renderers import TrainOnWhat
from hpcai.cookbook.supervised import train
from hpcai.cookbook.supervised.data import FromConversationFileBuilder
from hpcai.cookbook.supervised.types import ChatDatasetBuilderCommonConfig
import asyncio
import os

def build_config_blueprint() -> chz.Blueprint[train.Config]:
    model_name = "Qwen/Qwen3-4B"
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=model_name,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=32,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    dataset = chat_datasets.NoRobotsBuilder(common_config=common_config)
    if 0:  # To swap in your own dataset:
        dataset = FromConversationFileBuilder(
            common_config=common_config, file_path="/path/to/your/dataset.jsonl"
        )
        # ^^^ Create a dataset from a JSONL file in the same format as
        # hpcai.cookbook/example_data/conversations.jsonl
    return chz.Blueprint(train.Config).apply(
        {
            "log_path": "/tmp/hpcai-examples/sl_basic",
            "model_name": model_name,
            "dataset_builder": dataset,
            "learning_rate": 2e-4,
            "lr_schedule": "linear",
            "num_epochs": 1,
            "eval_every": 8
        }
    )


def main(config: train.Config):
    # Avoid clobbering log dir from your previous run:
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="ask")
    asyncio.run(train.main(config))


if __name__ == "__main__":
    blueprint = build_config_blueprint()
    blueprint.make_from_argv(sys.argv[1:])
    main(blueprint.make())
