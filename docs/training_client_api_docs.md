# TrainingClient API Reference

## Overview

The `TrainingClient` is used for training ML models with forward/backward passes and optimization. You typically get a `TrainingClient` by calling `service_client.create_lora_training_client()` or `service_client.create_training_client_from_state()`.


**Base URL**: `https://dev.hpc-ai.com/finetunesdk`
---

## Table of Contents

- [forward()](#forward)
- [forward_backward()](#forward_backward)
- [optim_step()](#optim_step)
- [get_info()](#get_info)
- [get_tokenizer()](#get_tokenizer)
- [save_state()](#save_state)
- [load_state()](#load_state)
- [save_weights_for_sampler()](#save_weights_for_sampler)
- [unload_model()](#unload_model)

---

## forward()

### Description

Performs a **forward pass** on the model **without computing gradients**. This method sends input data and loss-fn type (currently only supports "cross_entropy") to the training service and returns an asynchronous handle (`APIFuture`) for retrieving the `ForwardBackwardOutput`.

### Signature

```python
def forward(
    data: List[types.Datum],
    loss_fn: str = "cross_entropy"
) -> APIFuture[ForwardBackwardOutput]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `List[types.Datum]` | *required* | A list of `Datum` objects (from `hpcai.types.Datum`). Each `Datum` contains:<br>- **model_input**: The model input, which is input token ids.<br>- **loss_fn_inputs**: A dictionary of tensors required by the specified loss function. For `cross_entropy` loss, we need the `"weights"` (loss mask to mask per token loss) and `"target_tokens"` (the shifted labels matching the shape of weights and model_input). |
| `loss_fn` | `str` | `"cross_entropy"` | The name of the loss function to be used during the forward pass (currently only supports `"cross_entropy"`). |

### Returns

**APIFuture[ForwardBackwardOutput]**: An asynchronous handle that resolves to `ForwardBackwardOutput` when `.result()` is called.

For `"cross_entropy"` loss, the handle returns a `ForwardBackwardOutput` object that contains:

- **loss_fn_outputs**: A dictionary of log probability of each token and masked elementwise_loss, both of the same shape as the model_input
- **metrics**: A dictionary of training related metrics, including the mean training loss, the number of examples in the batch and the current step number

### Example

```python
from hpcai import types

data = [
    types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 4, 5]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=[2, 3, 4, 5, 6],
                dtype="int64",
                shape=[5]
            ),
            "weights": types.TensorData(
                data=[1.0, 1.0, 1.0, 1.0, 1.0],
                dtype="float32",
                shape=[5]
            )
        }
    )
]

out = training_client.forward(data, loss_fn="cross_entropy")
res = out.result()

print(res)
```

**Expected Output:**

```
ForwardBackwardOutput(loss_fn_output_type='cross_entropy', loss_fn_outputs=[{'logprobs': TensorData(data=[-8.9375, -10.375, -1.2734375, -0.65234375, -1.0], dtype='float32', shape=[5]), 'elementwise_loss': TensorData(data=[8.9375, 10.375, 1.2734375, 0.65234375, 1.0], dtype='float32', shape=[5])}], metrics={'loss:mean': 4.447656154632568, 'num_examples:sum': 1.0, 'step:max': 0.0})
```

---

## forward_backward()

### Description

Performs a forward–backward pass on the model. This call invokes loss.backward(), and gradients will accumulate across multiple forward_backward calls until optim_step is invoked. This method sends input data and loss-fn type (currently only supports "cross_entropy") to the training service and returns an asynchronous handle (APIFuture) for retrieving the ForwardBackwardOutput.

### Signature

```python
def forward_backward(
    data: List[types.Datum],
    loss_fn: str = "cross_entropy"
) -> APIFuture[ForwardBackwardOutput]
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data` | `List[types.Datum]` | *required* | A list of `Datum` objects (from `hpcai.types.Datum`). Each `Datum` contains:<br>- **model_input**: The model input, which is input token ids.<br>- **loss_fn_inputs**: A dictionary of tensors required by the specified loss function. For `cross_entropy` loss, we need the `"weights"` (loss mask to mask per token loss) and `"target_tokens"` (the shifted labels matching the shape of weights and model_input). |
| `loss_fn` | `str` | `"cross_entropy"` | The name of the loss function to be used during the forward pass (currently only supports `"cross_entropy"`). |

### Returns

An asynchronous handle that resolves to `ForwardBackwardOutput` when `.result()` is called.

For `"cross_entropy"` loss, the handle returns a `ForwardBackwardOutput` object that contains:

- **loss_fn_outputs**: A dictionary of log probability of each token and masked elementwise_loss, both of the same shape as the model_input
- **metrics**: A dictionary of training related metrics, including the mean training loss, the number of examples in the batch and the current step number

### Example

```python
from hpcai import types

data = [
    types.Datum(
        model_input=types.ModelInput.from_ints([1, 2, 3, 4, 5]),
        loss_fn_inputs={
            "target_tokens": types.TensorData(
                data=[2, 3, 4, 5, 6],
                dtype="int64",
                shape=[5]
            ),
            "weights": types.TensorData(
                data=[1.0, 1.0, 1.0, 1.0, 1.0],
                dtype="float32",
                shape=[5]
            )
        }
    )
]

out = training_client.forward_backward(data, loss_fn="cross_entropy")
res = out.result()

print(res)
```

**Expected Output:**

```
ForwardBackwardOutput(loss_fn_output_type='cross_entropy', loss_fn_outputs=[{'logprobs': TensorData(data=[-8.9375, -10.375, -1.2734375, -0.65234375, -1.0], dtype='float32', shape=[5]), 'elementwise_loss': TensorData(data=[8.9375, 10.375, 1.2734375, 0.65234375, 1.0], dtype='float32', shape=[5])}], metrics={'loss:mean': 4.447656154632568, 'num_examples:sum': 1.0, 'step:max': 0.0})
```

---

## optim_step()

### Description

Executes a single optimization step using the Adam optimizer and returns an asynchronous handle (APIFuture) for retrieving the OptimStepResponse. This call applies the accumulated gradients to update the model parameters based on the provided Adam hyperparameters.

### Signature

```python
def optim_step(
    adam_params: types.AdamParams
) -> APIFuture[OptimStepResponse]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `adam_params` | `types.AdamParams` | Configuration of the Adam optimizer, including:<br>- **learning_rate** (`float`): Step size for parameter updates. Default to 0.0001.<br>- **beta1** (`float`): Exponential decay rate for the first-moment estimates. Default to 0.9.<br>- **beta2** (`float`): Exponential decay rate for the second-moment estimates. Default to 0.95.<br>- **eps** (`float`): Small constant for numerical stability. Default to 1e-12. |

### Returns

Returns an `OptimStepResponse` object containing optimization metrics:

- **step:max**: The current step number
- **learning_rate:mean**: The learning rate used in this step
- **recent_loss:mean**: Mean training loss

### Example

```python
res = training_client.optim_step(adam_params=types.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.95, eps=1e-12)).result()

print(res)
```

**Expected Output:**

```
OptimStepResponse(metrics={'step:max': 1.0, 'learning_rate:mean': 0.0001, 'recent_loss:mean': 4.447656154632568})
```

---

## get_info()

### Description

Retrieves metadata and configuration details about the current training session. This includes model_id (run id), model architecture information, LoRA settings.

### Signature

```python
def get_info() -> GetInfoResponse
```

### Parameters

None

### Returns

Returns a `GetInfoResponse` object containing fields such as:

- **type**: The response type, usually "get_info"
- **model_data**: A `ModelData` object including architecture and model name
- **model_id**: Unique identifier of the model/run
- **is_lora**: Boolean indicating whether LoRA fine-tuning is enabled
- **lora_rank**: The LoRA rank if LoRA is enabled
- **model_name**: The base model name

### Example

```python
info = training_client.get_info()

print(info)
```

**Expected Output:**

```
GetInfoResponse(type='get_info', model_data=ModelData(arch='', model_name='Qwen/Qwen3-8B'), model_id='e5c88495-46e9-43df-9bf8-3185aceaa222', is_lora=True, lora_rank=16, model_name='Qwen/Qwen3-8B')
```

---

## get_tokenizer()

### Description

Returns a transformers pre-trained tokenizer object according to the loaded model of the current training run.

### Signature

```python
def get_tokenizer() -> PreTrainedTokenizer
```

### Parameters

None

### Returns

A `PreTrainedTokenizer` object from the Hugging Face Transformers library.

### Example

```python
tokenizer = training_client.get_tokenizer()

print(tokenizer.encode("Hello, world!"))
```

**Expected Output:**

```
[9707, 11, 1879, 0]
```

---

## save_state()

### Description

Save the trainable weights (e.g., if LoRA is enabled, only the weights of the LoRA adapter would be saved) of the loaded model with a user specified checkpoint_id for resuming training and returns an asynchronous handle (APIFuture) for retrieving the OptimStepResponse.

### Signature

```python
def save_state(
    name: str
) -> APIFuture[SaveWeightsResponse]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | The user specified checkpoint_id, which will be used as an identifier of the saved weights. |

### Returns

An asynchronous handle (APIFuture) for retrieving the `SaveWeightsResponse` object, which contains the url path of the saved weights in the format: `"hpcai://{model_id}/weights/{checkpoint_id}_training"`

### Example

```python
res = training_client.save_state("initial").result()

print(res)
```

**Expected Output:**

```
SaveWeightsResponse(path='hpcai://214c086a-f75b-49e1-9b1a-62ce2670baa7/weights/initial_training', type='save_weights')
```

---

## load_state()

### Description

Load the trainable weights (e.g., if LoRA is enabled, only the weights of the LoRA adapter would be loaded) of the user specified checkpoint by checkpoint_id and returns an asynchronous handle (APIFuture) for retrieving the OptimStepResponse.

### Signature

```python
def load_state(
    path: str
) -> APIFuture[LoadWeightsResponse]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | The url path of a saved checkpoint in the format: `"hpcai://{model_id}/weights/{checkpoint_id}_sampler"` (for inference) or in the format: `"hpcai://{model_id}/weights/{checkpoint_id}_training"` (for resume training) |

### Returns

An asynchronous handle (APIFuture) for retrieving the `LoadWeightsResponse` object, which contains the url path of the loaded weights if succeed.

### Example

```python
res = training_client.load_state('hpcai://e8b29733-9efa-476a-b62b-40b8f9c6d999/weights/initial_training').result()

print(res)
```

**Expected Output:**

```
LoadWeightsResponse(path='hpcai://e8b29733-9efa-476a-b62b-40b8f9c6d999/weights/initial_training', type='load_weights')
```

---

## save_weights_for_sampler()

### Description

Save the trainable weights (e.g., if LoRA is enabled, only the weights of the LoRA adapter would be saved) of the loaded model with a user specified checkpoint_id for inference and returns an asynchronous handle (APIFuture) for retrieving the SaveWeightsForSamplerResponse.

### Signature

```python
def save_weights_for_sampler(
    name: str
) -> APIFuture[SaveWeightsForSamplerResponse]
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | `str` | The user specified checkpoint_id, which will be used as an identifier of the saved weights. |

### Returns

An asynchronous handle (APIFuture) for retrieving the `SaveWeightsForSamplerResponse` object, which contains the url path of the saved weights in the format: `"hpcai://{model_id}/weights/{checkpoint_id}_sampler"`

### Example

```python
res = training_client.save_weights_for_sampler("initial").result()

print(res)
```

**Expected Output:**

```
SaveWeightsForSamplerResponse(path='hpcai://c857f372-1979-43e4-8afe-d0546ce38d98/sampler_weights/initial_sampler', type='save_weights_for_sampler')
```

---

## unload_model()

### Description

Signal the stop of the current training session and release resources such as GPUs. Please note that this API doesn't automatically save the model weights. It won't affect any previously saved checkpoints, you can still download them after model unload.

### Signature

```python
def unload_model() -> APIFuture[UnloadModelResponse]
```

### Parameters

None

### Returns

An asynchronous handle (APIFuture) for retrieving the `UnloadModelResponse` object, which contains the model id of the unloaded model if succeed.

### Example

```python
res = training_client.unload_model().result()

print(res)

res = rest_client.list_checkpoints(training_client.model_id).result()

print(res)
```

**Expected Output:**

```
UnloadModelResponse(model_id='c857f372-1979-43e4-8afe-d0546ce38d98', type='unload_model')
CheckpointsListResponse(checkpoints=[Checkpoint(checkpoint_id='initial_sampler', checkpoint_type='sampler', time=datetime.datetime(2025, 11, 28, 1, 58, 36, tzinfo=TzInfo(0)), checkpoint_path='hpcai://e8b29733-9efa-476a-b62b-40b8f9c6d999/sampler_weights/initial_sampler'), Checkpoint(checkpoint_id='initial_training', checkpoint_type='training', time=datetime.datetime(2025, 11, 28, 1, 58, 25, tzinfo=TzInfo(0)), checkpoint_path='hpcai://e8b29733-9efa-476a-b62b-40b8f9c6d999/weights/initial_training')])
```

---

## Related Documentation

- [ServiceClient API Reference](./service_client_api_docs.md)
- [RestClient API Reference](./rest_client_api_docs.md)
