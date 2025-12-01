# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

from enum import Enum


class ClientConnectionPoolType(Enum):
    SAMPLE = "sample"
    TRAIN = "train"
    RETRIEVE_PROMISE = "retrieve_promise"
    TELEMETRY = "telemetry"
