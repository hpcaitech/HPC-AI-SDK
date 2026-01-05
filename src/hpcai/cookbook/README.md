# Cookbook Usage Examples

# Test Supervised Learning

Currently, we support supervised training and rl training with `importance sampling` loss.

## test chat_sl train.py
```shell
cd $HOME/HPC-AI-SDK/src/hpcai/cookbook/recipes/chat_sl

python train.py \
    base_url="https://dev.hpc-ai.com/finetunesdk" \
    api_key="XXXXX" \
    model_name=Qwen/Qwen3-4B \
    dataset=no_robots \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    eval_every=20 \
    save_every=20 \
    wandb_project=cookbook_sl
```
<img width="3367" height="922" alt="image" src="https://github.com/user-attachments/assets/c9bc1bb6-f448-4da7-a005-382b74c7a9cb" />

## Test sl_basic.py
```shell
cd $HOME/HPC-AI-SDK/src/hpcai/cookbook/recipes

python sl_basic.py \
    base_url="https://dev.hpc-ai.com/finetunesdk" \
    api_key="XXXXX" \
    model_name=Qwen/Qwen3-4B \
    wandb_project=cookbook_sl_basic
```

## Test sl_loop.py
```shell
cd $HOME/HPC-AI-SDK/src/hpcai/cookbook/recipes

python sl_loop.py \
    base_url="https://dev.hpc-ai.com/finetunesdk" \
    api_key="XXXXX" \
    model_name=Qwen/Qwen3-4B \
    learning_rate=5e-4 \
    batch_size=64 \
    lora_rank=64 \
    save_every=20
```
<img width="607" height="732" alt="image" src="https://github.com/user-attachments/assets/62ef9de5-f2ec-4461-9e97-5da01e3393ce" />

## Test math/train.py
```shell
cd $HOME/HPC-AI-SDK/src/hpcai/cookbook/recipes/math_rl

python train.py \
    base_url="https://dev.hpc-ai.com/finetunesdk" \
    api_key="XXX" \
    model_name=Qwen/Qwen3-1.7B \
    env=gsm8k \
    learning_rate=1e-6 \
    groups_per_batch=2 \    # to avoid 429 RateLimitError
    lora_rank=64 \
    eval_every=20 \
    save_every=20 \
    wandb_project=cookbook_math_rl
```
<img width="2145" height="1495" alt="image" src="https://github.com/user-attachments/assets/c7cd6ea1-8629-478b-9eea-41a7303a0809" />
<img width="616" height="1320" alt="image" src="https://github.com/user-attachments/assets/fe822926-5a5d-4cd0-84db-33f10c9c2d82" />

