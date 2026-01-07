<h1 align="center">HPC-AI Python SDK</h1>

## Overview

The HPC-AI Python SDK provides a powerful interface for distributed GPU training and fine-tuning on HPC-AI's cloud infrastructure.

## Installation

we recommend using conda to install the SDK.

```bash
conda create -n hpcai python=3.12 -y
conda activate hpcai
pip install hpcai
```

## Quick Start

```python
from hpcai import ServiceClient, TrainingClient

# Initialize the service client
client = ServiceClient(
    base_url="https://www.hpc-ai.com/finetunesdk",
    api_key="your-api-key"
)

# Create a training client for LoRA fine-tuning
training_client = client.create_lora_training_client(
    base_model="Qwen/Qwen2.5-7B",
    rank=8,
    seed=42
)
```



### Path Protocol

The SDK uses the `hpcai://` protocol for model and checkpoint paths:

```python
model_path = "hpcai://run-123/training/checkpoint-001"
```


### Environment Variables

Configure the SDK using these environment variables:

- `HPCAI_API_KEY` - Your API key
- `HPCAI_BASE_URL` - API endpoint (default: https://www.hpc-ai.com/finetunesdk)


## Features

- **Distributed Training**: Leverage HPC-AI's GPU cloud for efficient model training
- **LoRA Fine-tuning**: Memory-efficient fine-tuning with LoRA adapters
- **Async Support**: Full async/await support for concurrent operations
- **Type Safety**: Comprehensive type hints for better IDE support


## Usage Example

[A usage example for finetune "Qwen3-8B" model](./Usage.md).

## Cookbook

We provide a cookbook for you to use the SDK to train your models. Code can be found [here](./src/hpcai/cookbook).

Clone the repo to check more detail usage about the cookbook.
```
git clone https://github.com/hpcaitech/HPC-AI-SDK
cd HPC-AI-SDK/src/hpcai/cookbook
```

## Documentation

### API Reference

- [ServiceClient API Reference](./docs/service_client_api_docs.md) - Main entry point for creating clients and querying server capabilities
- [TrainingClient API Reference](./docs/training_client_api_docs.md) - Training operations including forward/backward passes and optimization
- [RestClient API Reference](./docs/rest_client_api_docs.md) - REST API operations for querying training runs and checkpoints

## Development

This repository uses `pre-commit` for basic formatting and hygiene checks.

```bash
pip install -r requirements-dev.txt
pre-commit install
pre-commit run -a
```

## Third-Party Notice

This SDK provides interoperability with components based on the Tinker project (Apache License 2.0).
Tinker is a trademark of its respective owner. This project is not affiliated with or endorsed by Thinking Machines Lab.

## License

Licensed under the Apache License, Version 2.0. See LICENSE file for details.
