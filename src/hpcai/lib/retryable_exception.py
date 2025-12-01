# Copyright 2025 Thinking Machines Lab
#
# Licensed under the Apache License, Version 2.0
#
# Modifications:
# - Adapted for HPC-AI cloud fine-tuning workflow
# Copyright © 2025 HPC-AI.COM

class RetryableException(Exception):
    def __init__(self, message: str):
        super().__init__(message)
