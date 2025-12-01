# ServiceClient API Reference

## Overview

The `ServiceClient` is the main entry point for interacting with the HPC-AI Fine-Tune SDK. It provides methods to discover server capabilities, create training clients, and manage REST API operations.

**Base URL**: `https://dev.hpc-ai.com/finetunesdk`

---

## Table of Contents

- [get_server_capabilities()](#get_server_capabilities)
- [create_lora_training_client()](#create_lora_training_client)
- [create_training_client_from_state()](#create_training_client_from_state)
- [create_training_client()](#create_training_client)
- [create_rest_client()](#create_rest_client)

---

## get_server_capabilities()

### Description

Queries the API server to discover what models and capabilities are available. Returns a list of supported models that can be used for training or inference operations.

This is typically the first method called to check server availability and discover available models before creating training clients.

### Signature

```python
def get_server_capabilities() -> GetServerCapabilitiesResponse
```

### Parameters

None

### Returns

**GetServerCapabilitiesResponse**: A response object containing:

- **supported_models** (`List[SupportedModel]`): A list of model information objects, where each `SupportedModel` contains:
  - **model_name** (`Optional[str]`): The name/identifier of the model

### Example

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
capabilities = service_client.get_server_capabilities()

print(f"Server supports {len(capabilities.supported_models)} models")
for model in capabilities.supported_models:
    if model.model_name:
        print(f"  - {model.model_name}")
```

### Additional Notes

- This API supports an async version: `get_server_capabilities_async()`

---

## create_lora_training_client()

### Description

Creates a new LoRA-adapted model instance on the server and returns a `TrainingClient` that can be used to train the model.

The method performs the following operations:
1. Validates the initialization input
2. Creates a new model instance on the server with LoRA adapters applied
3. Returns a `TrainingClient` bound to the newly created model

### Signature

```python
def create_lora_training_client(
    base_model: str,
    rank: int = 32,
    seed: int | None = None,
    train_mlp: bool = True,
    train_attn: bool = True,
    train_unembed: bool = True,
) -> TrainingClient
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_model` | `str` | *required* | The name or identifier of the base model to fine-tune. Must be a model supported by the server (check with `get_server_capabilities()`). |
| `rank` | `int` | `32` | The rank (dimension) used in LoRA adapters. Higher ranks allow more capacity but increase trainable parameters. Common values range from 8 to 128. Typical values:<br>- Small models (< 1B): 8-16<br>- Medium models (1B-7B): 16-32<br>- Large models (> 7B): 32-64 |
| `seed` | `int \| None` | `None` | Optional random seed for reproducible LoRA weight initialization. If provided, the LoRA adapters will be initialized deterministically. |
| `train_mlp` | `bool` | `True` | Whether to apply LoRA adapters to MLP (Multi-Layer Perceptron) layers, including gate, up, and down projection layers. |
| `train_attn` | `bool` | `True` | Whether to apply LoRA adapters to attention layers, including query (q_proj), key (k_proj), value (v_proj), and output (o_proj) projection matrices. |
| `train_unembed` | `bool` | `True` | Whether to apply LoRA adapters to the unembedding layer (language modeling head, lm_head). |

### Returns

**TrainingClient**: A client instance bound to the newly created LoRA model.

### Example

**Basic usage with default LoRA configuration:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen3-8B"
)

print(f"Created model ID {training_client.model_id}")
```

**Custom LoRA configuration for attention-only fine-tuning:**

```python
training_client = service_client.create_lora_training_client(
    base_model="Qwen/Qwen2.5-7B-Instruct",
    rank=64,
    seed=42,
    train_mlp=False,
    train_attn=True,  # only fine tune on attention layers
    train_unembed=False
)
```

### Additional Notes

- This API supports an async version: `create_lora_training_client_async()`

---

## create_training_client_from_state()

### Description

Creates a training session from a previously saved checkpoint, which allows you to resume training from a trained checkpoint. It automatically retrieves the training run metadata (base model, LoRA rank) from the checkpoint path, creates a new LoRA model instance with matching configuration, and loads the saved weights into it.

### Signature

```python
def create_training_client_from_state(path: str) -> TrainingClient
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | A URI path to the checkpoint. Must follow the format:<br>- Training checkpoint: `"hpcai://{training_run_id}/weights/{checkpoint_id}"`<br>- Sampler checkpoint: `"hpcai://{training_run_id}/sampler_weights/{checkpoint_id}"` |

### Returns

**TrainingClient**: A client instance bound to the restored model with checkpoint weights loaded.

### Example

**Resuming training from a saved checkpoint:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
CHECKPOINT_PATH = "hpcai://{model_id}/weights/step_0010"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
training_client = service_client.create_training_client_from_state(CHECKPOINT_PATH)

print(f"Restored model ID: {training_client.model_id}")

# Continue training
training_data = [...]  # Your training data
fwd_bwd = training_client.forward_backward(training_data, loss_fn="cross_entropy")
optim = training_client.optim_step(hpcai.AdamParams(learning_rate=1e-4))

fwd_bwd.result()
optim.result()

# Clean up
training_client.unload_model().result()
```

**Getting checkpoint path from RestClient and restoring:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# List checkpoints for a training run
training_run_id = "eb78693b-380d-40f8-a709-ffe0da185718"
checkpoints_response = rest_client.list_checkpoints(training_run_id).result()

# Find the latest checkpoint
if checkpoints_response.checkpoints:
    latest_checkpoint = checkpoints_response.checkpoints[-1]
    checkpoint_path = latest_checkpoint.checkpoint_path
    print(f"Restoring from: {checkpoint_path}")
    
    # Restore training client
    training_client = service_client.create_training_client_from_state(checkpoint_path)
    print(f"Restored model ID: {training_client.model_id}")
```

### Additional Notes

- This API supports an async version: `create_training_client_from_state_async()`

---

## create_training_client()

### Description

Creates a `TrainingClient` instance for an existing or new model. This method creates a `TrainingClient` wrapper around a model identified by `model_id`. Use this method when you already have a `model_id` from a previous session.

### Signature

```python
def create_training_client(
    model_id: types.ModelID | None = None
) -> TrainingClient
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_id` | `types.ModelID \| None` | `None` | Optional unique identifier of an existing model on the server. If provided, the `TrainingClient` will be bound to this model. If `None`, the client is created without a bound model. |

### Returns

**TrainingClient**: A client instance that can be used for training operations. If `model_id` is provided, the client is bound to that model. If `model_id` is `None`, the client has no bound model and requires loading a checkpoint or creating a model before use.

### Example

**Creating a client for an existing model from a previous session:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"

# You have a model_id from a previous training session
existing_model_id = "eb78693b-380d-40f8-a709-ffe0da185718"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
training_client = service_client.create_training_client(model_id=existing_model_id)

print(f"Client bound to model: {training_client.model_id}")

# Use the client for training
training_data = [...]  # Your training data
fwd_bwd = training_client.forward_backward(training_data, loss_fn="cross_entropy")
optim = training_client.optim_step(hpcai.AdamParams(learning_rate=1e-4))

fwd_bwd.result()
optim.result()
```

**Creating a client without model_id, then loading a checkpoint:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
CHECKPOINT_PATH = "hpcai://eb78693b-380d-40f8-a709-ffe0da185718/weights/step_0010"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)

# Create client without model_id
training_client = service_client.create_training_client(model_id=None)

# Load checkpoint to bind the model
load_future = training_client.load_state(CHECKPOINT_PATH)
load_future.result()

print(f"Model loaded: {training_client.model_id}")

# Now you can use it for training
training_data = [...]  # Your training data
result = training_client.forward_backward(training_data, loss_fn="cross_entropy")
result.result()
```

---

## create_rest_client()

### Description

Creates a `RestClient` instance for REST API operations. This method creates a `RestClient` that provides access to various REST endpoints for querying training runs, checkpoints, and other metadata.

### Signature

```python
def create_rest_client() -> RestClient
```

### Parameters

None

### Returns

**RestClient**: Client for REST API operations like listing checkpoints and metadata.

### Example

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
TRAINING_RUN_ID = "eb78693b-380d-40f8-a709-ffe0da185718"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# Use RestClient methods
checkpoints_response = rest_client.list_checkpoints(TRAINING_RUN_ID).result()
training_run = rest_client.get_training_run(TRAINING_RUN_ID).result()
```

---

## Related Documentation

- [RestClient API Reference](./rest_client_api_docs.md)
- [TrainingClient API Reference](./training_client_api_docs.md)

