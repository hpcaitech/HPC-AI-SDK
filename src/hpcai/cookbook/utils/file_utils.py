# Copyright 2025 Thinking Machines Lab
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM
import json


def read_jsonl(path: str) -> list[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
