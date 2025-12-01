# RestClient API Reference

## Overview

The `RestClient` provides REST API operations for querying training runs, checkpoints, and other metadata. You typically get a `RestClient` by calling `service_client.create_rest_client()`.

**Base URL**: `https://dev.hpc-ai.com/finetunesdk`

---

## Table of Contents

- [list_training_runs()](#list_training_runs)
- [get_training_run()](#get_training_run)
- [get_training_run_by_checkpoint_path()](#get_training_run_by_checkpoint_path)
- [list_checkpoints()](#list_checkpoints)
- [download_checkpoint_archive()](#download_checkpoint_archive)
- [download_checkpoint_archive_by_checkpoint_path()](#download_checkpoint_archive_by_checkpoint_path)
- [delete_checkpoint()](#delete_checkpoint)
- [delete_checkpoint_by_checkpoint_path()](#delete_checkpoint_by_checkpoint_path)

---

## list_training_runs()

### Description

List training runs with pagination support. This method retrieves a paginated list of training runs from the server. Each training run represents a model instance that has been created for training, including both active and completed training sessions.

### Signature

```python
def list_training_runs(
    limit: int = 20,
    offset: int = 0
) -> ConcurrentFuture[TrainingRunsResponse]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | `int` | `20` | Maximum number of training runs to return in this request. Must be a positive integer. |
| `offset` | `int` | `0` | Number of training runs to skip before returning results. Used for pagination. Default is 0 (start from the beginning). For subsequent pages, set offset to the cumulative number of items already retrieved. |

### Returns

**TrainingRunsResponse**: A response object that contains:

- **training_runs** (`list[TrainingRun]`): List of training run objects, each containing:
  - **training_run_id** (`str`): Unique identifier for the training run
  - **base_model** (`str`): Base model name
  - **model_owner** (`str`): Owner/creator identifier
  - **is_lora** (`bool`): Whether this is a LoRA model
  - **corrupted** (`bool`): Whether the model is in a corrupted state
  - **lora_rank** (`int | None`): LoRA rank if applicable, None otherwise
  - **last_request_time** (`datetime`): Timestamp of last request
  - **last_checkpoint** (`Checkpoint | None`): Most recent training checkpoint
  - **last_sampler_checkpoint** (`Checkpoint | None`): Most recent sampler checkpoint
- **cursor** (`Cursor`): Pagination metadata containing:
  - **offset** (`int`): The offset used in this request
  - **limit** (`int`): The limit used in this request
  - **total_count** (`int`): Total number of training runs available

### Example

**Basic usage to list training runs:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# List first 10 training runs
response = rest_client.list_training_runs(limit=10, offset=0).result()

print(f"Found {len(response.training_runs)} training runs")
print(f"Total available: {response.cursor.total_count}")

for run in response.training_runs:
    print(f"  - {run.training_run_id}: {run.base_model}")
    if run.is_lora:
        print(f"    LoRA rank: {run.lora_rank}")
```

**Finding training runs for a specific base model:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
TARGET_MODEL = "Qwen/Qwen3-8B"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# List training runs and filter
response = rest_client.list_training_runs(limit=100, offset=0).result()

matching_runs = [
    run for run in response.training_runs
    if run.base_model == TARGET_MODEL
]

print(f"Found {len(matching_runs)} training runs for {TARGET_MODEL}:")
for run in matching_runs:
    print(f"  - {run.training_run_id} (LoRA rank: {run.lora_rank})")
```

### Additional Notes

- This API supports the async version: `list_training_runs_async()`

---

## get_training_run()

### Description

Get detailed information for a specific training run by model_id. This method retrieves comprehensive metadata about a single training run.

### Signature

```python
def get_training_run(
    training_run_id: types.ModelID
) -> ConcurrentFuture[TrainingRun]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_run_id` | `types.ModelID` | The unique identifier of the training run to query. Must match an existing training run on the server. |

### Returns

**TrainingRun**: An object that contains:

- **training_run_id** (`str`): Unique identifier for the training run
- **base_model** (`str`): Base model name
- **model_owner** (`str`): Owner/creator identifier
- **is_lora** (`bool`): Whether this is a LoRA model
- **corrupted** (`bool`): Whether the model is in a corrupted state
- **lora_rank** (`int | None`): LoRA rank if applicable, None otherwise
- **last_request_time** (`datetime`): Timestamp of last request
- **last_checkpoint** (`Checkpoint | None`): Most recent training checkpoint
- **last_sampler_checkpoint** (`Checkpoint | None`): Most recent sampler checkpoint

### Example

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
TRAINING_RUN_ID = "eb78693b-380d-40f8-a709-ffe0da185718"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# Get training run information
training_run = rest_client.get_training_run(TRAINING_RUN_ID).result()

print(f"Training Run ID: {training_run.training_run_id}")
print(f"Base Model: {training_run.base_model}")
print(f"Is LoRA: {training_run.is_lora}")
if training_run.is_lora:
    print(f"LoRA Rank: {training_run.lora_rank}")
print(f"Owner: {training_run.model_owner}")
print(f"Last Request: {training_run.last_request_time}")
print(f"Corrupted: {training_run.corrupted}")
```

### Additional Notes

- This API supports the async version: `get_training_run_async(training_run_id: types.ModelID)`

---

## get_training_run_by_checkpoint_path()

### Description

Get detailed information for a specific training run by checkpoint_path. This method retrieves comprehensive metadata about a single training run.

### Signature

```python
def get_training_run_by_checkpoint_path(
    checkpoint_path: str
) -> ConcurrentFuture[TrainingRun]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | `str` | A checkpoint path using the hpcai:// protocol. Must follow the format:<br>- Training checkpoint: `"hpcai://{training_run_id}/weights/{checkpoint_id}"`<br>- Sampler checkpoint: `"hpcai://{training_run_id}/sampler_weights/{checkpoint_id}"` |

### Returns

**TrainingRun**: An object that contains:

- **training_run_id** (`str`): Unique identifier for the training run
- **base_model** (`str`): Base model name
- **model_owner** (`str`): Owner/creator identifier
- **is_lora** (`bool`): Whether this is a LoRA model
- **corrupted** (`bool`): Whether the model is in a corrupted state
- **lora_rank** (`int | None`): LoRA rank if applicable, None otherwise
- **last_request_time** (`datetime`): Timestamp of last request
- **last_checkpoint** (`Checkpoint | None`): Most recent training checkpoint
- **last_sampler_checkpoint** (`Checkpoint | None`): Most recent sampler checkpoint

### Example

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
CHECKPOINT_PATH = "hpcai://eb78693b-380d-40f8-a709-ffe0da185718/weights/step_0010"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# Get training run info from checkpoint path
training_run = rest_client.get_training_run_by_checkpoint_path(CHECKPOINT_PATH).result()

print(f"Training Run ID: {training_run.training_run_id}")
print(f"Base Model: {training_run.base_model}")
print(f"Is LoRA: {training_run.is_lora}")
if training_run.is_lora:
    print(f"LoRA Rank: {training_run.lora_rank}")
```

### Additional Notes

- This API supports the async version: `get_training_run_by_checkpoint_path_async(checkpoint_path: str)`

---

## list_checkpoints()

### Description

List all available checkpoints for a specific training run. This method retrieves a complete list of checkpoints saved for a given training run. Checkpoints can be of two types: "training" checkpoints (containing model weights saved during training) and "sampler" checkpoints (optimized for inference).

### Signature

```python
def list_checkpoints(
    training_run_id: types.ModelID
) -> ConcurrentFuture[CheckpointsListResponse]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_run_id` | `types.ModelID` | The unique identifier of the training run to query. Must match an existing training run on the server. |

### Returns

**CheckpointsListResponse**: A response object that contains:

- **checkpoints** (`list[Checkpoint]`): List of checkpoint objects, each containing:
  - **checkpoint_id** (`str`): Unique identifier for the checkpoint
  - **checkpoint_type** (`CheckpointType`): Either "training" or "sampler"
    - **"training"**: Checkpoint containing full trained LoRA adapter weights for resuming training
    - **"sampler"**: Checkpoint optimized for inference/sampling, smaller size, faster loading
  - **time** (`datetime`): Timestamp when the checkpoint was created (with timezone information, typically UTC)
  - **checkpoint_path** (`str`): Full checkpoint path. Format: `"hpcai://{training_run_id}/{type}/{checkpoint_id}"`

### Example

**Basic usage to list all checkpoints:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
TRAINING_RUN_ID = "eb78693b-380d-40f8-a709-ffe0da185718"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# List all checkpoints
checkpoints_response = rest_client.list_checkpoints(TRAINING_RUN_ID).result()

print(f"Found {len(checkpoints_response.checkpoints)} checkpoints:")
for checkpoint in checkpoints_response.checkpoints:
    print(f"  - {checkpoint.checkpoint_type}: {checkpoint.checkpoint_id}")
    print(f"    Time: {checkpoint.time}")
    print(f"    Path: {checkpoint.checkpoint_path}")
```

**Finding the latest checkpoint:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
TRAINING_RUN_ID = "eb78693b-380d-40f8-a709-ffe0da185718"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

checkpoints_response = rest_client.list_checkpoints(TRAINING_RUN_ID).result()

if checkpoints_response.checkpoints:
    # Sort by time (most recent first) and get the latest
    sorted_checkpoints = sorted(
        checkpoints_response.checkpoints,
        key=lambda cp: cp.time,
        reverse=True
    )
    latest_checkpoint = sorted_checkpoints[0]
    
    print(f"Latest checkpoint: {latest_checkpoint.checkpoint_id}")
    print(f"  Type: {latest_checkpoint.checkpoint_type}")
    print(f"  Time: {latest_checkpoint.time}")
    print(f"  Path: {latest_checkpoint.checkpoint_path}")
    
    # Use it to restore training
    training_client = service_client.create_training_client_from_state(
        latest_checkpoint.checkpoint_path
    )
    print(f"Restored model ID: {training_client.model_id}")
```

### Additional Notes

- This API supports the async version: `list_checkpoints_async(training_run_id: types.ModelID)`
- Currently only support saving LoRA adapter weights, we will support saving optimizer state very soon!

---

## download_checkpoint_archive()

### Description

Download a checkpoint as a compressed tar.gz archive by model_id and checkpoint_id. This method downloads a complete checkpoint archive from the server, containing all model weights, configuration, and metadata needed to run the trained model locally. The archive is returned as raw bytes, which you can save to disk or process directly.

### Signature

```python
def download_checkpoint_archive(
    training_run_id: types.ModelID,
    checkpoint_id: str
) -> ConcurrentFuture[bytes]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_run_id` | `types.ModelID` | The unique identifier of the training run to query. Must match an existing training run on the server. |
| `checkpoint_id` | `str` | The identifier of the checkpoint to download. This is the checkpoint_id field from list_checkpoints() results. |

### Returns

**bytes**: A series of bytes that contains the checkpoint archive data. The archive is a tar.gz compressed file containing:

- Model weights (adapter_model.bin or adapter_model.safetensors for LoRA)
- Configuration files (adapter_config.json, tokenizer files)
- Any other checkpoint metadata

### Example

**Basic usage to download and save a checkpoint:**

```python
import hpcai
from pathlib import Path

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
TRAINING_RUN_ID = "eb78693b-380d-40f8-a709-ffe0da185718"
CHECKPOINT_ID = "step_0010"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# Download checkpoint archive
archive_data = rest_client.download_checkpoint_archive(
    training_run_id=TRAINING_RUN_ID,
    checkpoint_id=CHECKPOINT_ID
).result()

# Save to local file
output_path = f"./checkpoints/checkpoint_{CHECKPOINT_ID}.tar.gz"
Path("./checkpoints").mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as f:
    f.write(archive_data)

print(f"Checkpoint downloaded to: {output_path}")
print(f"File size: {len(archive_data) / 1024 / 1024:.2f} MB")
```

**Downloading and extracting a checkpoint:**

```python
import hpcai
import tarfile
from pathlib import Path

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
TRAINING_RUN_ID = "eb78693b-380d-40f8-a709-ffe0da185718"
CHECKPOINT_ID = "final"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# Download checkpoint
archive_data = rest_client.download_checkpoint_archive(
    training_run_id=TRAINING_RUN_ID,
    checkpoint_id=CHECKPOINT_ID
).result()

# Save archive
archive_path = Path(f"./checkpoints/checkpoint_{CHECKPOINT_ID}.tar.gz")
archive_path.parent.mkdir(parents=True, exist_ok=True)
archive_path.write_bytes(archive_data)

# Extract archive
extract_dir = Path(f"./checkpoints/extracted_{CHECKPOINT_ID}")
extract_dir.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(extract_dir)

print(f"Checkpoint extracted to: {extract_dir}")
print(f"Contents: {list(extract_dir.iterdir())}")
```

### Additional Notes

- This API supports the async version: `download_checkpoint_archive_async(training_run_id: types.ModelID, checkpoint_id: str)`
- Downloading checkpoints consumes bandwidth and storage. We suggest consider downloading only when needed for local analysis or backup purposes.

---

## download_checkpoint_archive_by_checkpoint_path()

### Description

Download a checkpoint as a compressed tar.gz archive by checkpoint_path. This method downloads a complete checkpoint archive from the server, containing all model weights, configuration, and metadata needed to run the trained model locally. The archive is returned as raw bytes, which you can save to disk or process directly.

### Signature

```python
def download_checkpoint_archive_by_checkpoint_path(
    checkpoint_path: str
) -> ConcurrentFuture[bytes]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | `str` | A checkpoint path using the hpcai:// protocol. Must follow the format:<br>- Training checkpoint: `"hpcai://{training_run_id}/weights/{checkpoint_id}"`<br>- Sampler checkpoint: `"hpcai://{training_run_id}/sampler_weights/{checkpoint_id}"` |

### Returns

**bytes**: A series of bytes that contains the checkpoint archive data. The archive is a tar.gz compressed file containing:

- Model weights (adapter_model.bin or adapter_model.safetensors for LoRA)
- Configuration files (adapter_config.json, tokenizer files)
- Any other checkpoint metadata

### Example

**Basic usage to download and save a checkpoint:**

```python
import hpcai
from pathlib import Path

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
CHECKPOINT_PATH = "hpcai://eb78693b-380d-40f8-a709-ffe0da185718/weights/step_0010"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# Download checkpoint archive
archive_data = rest_client.download_checkpoint_archive_by_checkpoint_path(
    checkpoint_path=CHECKPOINT_PATH
).result()

# Save to local file
output_path = f"./checkpoints/{CHECKPOINT_PATH}.tar.gz"
Path("./checkpoints").mkdir(parents=True, exist_ok=True)
with open(output_path, "wb") as f:
    f.write(archive_data)

print(f"Checkpoint downloaded to: {output_path}")
print(f"File size: {len(archive_data) / 1024 / 1024:.2f} MB")
```

**Downloading and extracting a checkpoint:**

```python
import hpcai
import tarfile
from pathlib import Path

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
CHECKPOINT_PATH = "hpcai://eb78693b-380d-40f8-a709-ffe0da185718/weights/step_0010"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# Download checkpoint
archive_data = rest_client.download_checkpoint_archive_by_checkpoint_path(
    checkpoint_path=CHECKPOINT_PATH
).result()

# Save archive
archive_path = Path(f"./checkpoints/{CHECKPOINT_PATH}.tar.gz")
archive_path.parent.mkdir(parents=True, exist_ok=True)
archive_path.write_bytes(archive_data)

# Extract archive
extract_dir = Path(f"./checkpoints/extracted_{CHECKPOINT_PATH}")
extract_dir.mkdir(parents=True, exist_ok=True)
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(extract_dir)

print(f"Checkpoint extracted to: {extract_dir}")
print(f"Contents: {list(extract_dir.iterdir())}")
```

### Additional Notes

- This API supports the async version: `download_checkpoint_archive_by_checkpoint_path_async(checkpoint_path: str)`
- Downloading checkpoints consumes bandwidth and storage. We suggest consider downloading only when needed for local analysis or backup purposes.

---

## delete_checkpoint()

### Description

Permanently delete a checkpoint for a training run by model_id and checkpoint_id. This method permanently removes a checkpoint from the server storage. The deletion is irreversible. Use this method to free up storage space or remove checkpoints that are no longer needed.

### Signature

```python
def delete_checkpoint(
    training_run_id: types.ModelID,
    checkpoint_id: str
) -> ConcurrentFuture[None]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `training_run_id` | `types.ModelID` | The unique identifier of the training run to query. Must match an existing training run on the server. |
| `checkpoint_id` | `str` | The identifier of the checkpoint to download. This is the checkpoint_id field from list_checkpoints() results. |

### Returns

**None**

### Example

**Basic usage to delete a checkpoint:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
TRAINING_RUN_ID = "eb78693b-380d-40f8-a709-ffe0da185718"
CHECKPOINT_ID = "step_0010"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# WARNING: This permanently deletes the checkpoint
try:
    rest_client.delete_checkpoint(
        training_run_id=TRAINING_RUN_ID,
        checkpoint_id=CHECKPOINT_ID
    ).result()
    print(f"Checkpoint {CHECKPOINT_ID} deleted successfully")
except Exception as e:
    print(f"Failed to delete checkpoint: {e}")
```

**Safe deletion with backup check:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
TRAINING_RUN_ID = "eb78693b-380d-40f8-a709-ffe0da185718"
CHECKPOINT_ID = "step_0010"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# Check if checkpoint exists before deletion
checkpoints_response = rest_client.list_checkpoints(TRAINING_RUN_ID).result()
checkpoint_exists = any(
    cp.checkpoint_id == CHECKPOINT_ID
    for cp in checkpoints_response.checkpoints
)

if not checkpoint_exists:
    print(f"Checkpoint {CHECKPOINT_ID} does not exist")
else:
    print(f"Found checkpoint {CHECKPOINT_ID}")
    print("WARNING: This will permanently delete the checkpoint!")
    
    # Optional: Download backup before deletion
    # archive_data = rest_client.download_checkpoint_archive(
    #     training_run_id=TRAINING_RUN_ID,
    #     checkpoint_id=CHECKPOINT_ID
    # ).result()
    # with open(f"backup_{CHECKPOINT_ID}.tar.gz", "wb") as f:
    #     f.write(archive_data)
    # print("Backup downloaded")
    
    # Delete checkpoint
    rest_client.delete_checkpoint(
        training_run_id=TRAINING_RUN_ID,
        checkpoint_id=CHECKPOINT_ID
    ).result()
    print(f"Checkpoint {CHECKPOINT_ID} deleted")
```

### Additional Notes

- This API supports the async version: `delete_checkpoint_async(training_run_id: types.ModelID, checkpoint_id: str)`

---

## delete_checkpoint_by_checkpoint_path()

### Description

Permanently delete a checkpoint for a training run by checkpoint_path. This method permanently removes a checkpoint from the server storage. The deletion is irreversible. Use this method to free up storage space or remove checkpoints that are no longer needed.

### Signature

```python
def delete_checkpoint_by_checkpoint_path(
    checkpoint_path: str
) -> ConcurrentFuture[None]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `checkpoint_path` | `str` | A checkpoint path using the hpcai:// protocol. Must follow the format:<br>- Training checkpoint: `"hpcai://{training_run_id}/weights/{checkpoint_id}"`<br>- Sampler checkpoint: `"hpcai://{training_run_id}/sampler_weights/{checkpoint_id}"` |

### Returns

**None**

### Example

**Basic usage to delete a checkpoint:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
CHECKPOINT_PATH = "hpcai://eb78693b-380d-40f8-a709-ffe0da185718/weights/step_0010"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# WARNING: This permanently deletes the checkpoint
try:
    rest_client.delete_checkpoint_by_checkpoint_path(
        checkpoint_path=CHECKPOINT_PATH,
    ).result()
    print(f"Checkpoint {CHECKPOINT_PATH} deleted successfully")
except Exception as e:
    print(f"Failed to delete checkpoint: {e}")
```

**Safe deletion with backup check:**

```python
import hpcai

BASE_URL = "https://dev.hpc-ai.com/finetunesdk"
API_KEY = "your-api-key-here"
CHECKPOINT_PATH = "hpcai://eb78693b-380d-40f8-a709-ffe0da185718/weights/step_0010"

service_client = hpcai.ServiceClient(base_url=BASE_URL, api_key=API_KEY)
rest_client = service_client.create_rest_client()

# Check if checkpoint exists before deletion
checkpoints_response = rest_client.list_checkpoints(TRAINING_RUN_ID).result()
checkpoint_exists = any(
    cp.checkpoint_path == CHECKPOINT_PATH
    for cp in checkpoints_response.checkpoints
)

if not checkpoint_exists:
    print(f"Checkpoint {CHECKPOINT_PATH} does not exist")
else:
    print(f"Found checkpoint {CHECKPOINT_PATH}")
    print("WARNING: This will permanently delete the checkpoint!")
    
    # Delete checkpoint
    rest_client.delete_checkpoint_by_checkpoint_path(
        checkpoint_path=CHECKPOINT_PATH,
    ).result()
    print(f"Checkpoint {CHECKPOINT_PATH} deleted")
```

### Additional Notes

- This API supports the async version: `delete_checkpoint_by_checkpoint_path_async(checkpoint_path: str)`

---

## Related Documentation

- [ServiceClient API Reference](./service_client_api_docs.md)
- [TrainingClient API Reference](./training_client_api_docs.md)

