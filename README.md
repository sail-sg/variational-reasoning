# Variational Reasoning for Language Models

This is the official repository for the paper "Variational Reasoning for Language Models".

The repository currently includes data processing, training pipelines, and an evaluation suite. It's initialized from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [SkyThought](https://github.com/NovaSky-AI/SkyThought).

## Models and Datasets

| Final Reasoning Model $\pi_\theta$ | Backbone | Initial Reasoning Model $\pi_{\theta_0}$ | Variational Posterior $q_\phi$ | Final Training Dataset |
|:---------|:-----|:----|:------|:------|
| [🤗Variational-Reasoning-4B-Acc](https://huggingface.co/zhouxiangxin/Variational-Reasoning-4B-Acc) | Qwen3-4B-Base | [🤗Initial-Reasoning-4B](https://huggingface.co/zhouxiangxin/Initial-Reasoning-4B) | [🤗Variational-Posterior-PB-4B](https://huggingface.co/zhouxiangxin/Variational-Posterior-PB-4B) | [🤗Variational-Posterior-4B-Acc-mix](https://huggingface.co/datasets/zhouxiangxin/Variational-Posterior-4B-Acc-mix) |
| [🤗Variational-Reasoning-4B-GML](https://huggingface.co/zhouxiangxin/Variational-Reasoning-4B-GML) | Qwen3-4B-Base | [🤗Initial-Reasoning-4B](https://huggingface.co/zhouxiangxin/Initial-Reasoning-4B) | [🤗Variational-Posterior-PB-4B](https://huggingface.co/zhouxiangxin/Variational-Posterior-PB-4B) | [🤗Variational-Posterior-4B-GML-mix](https://huggingface.co/datasets/zhouxiangxin/Variational-Posterior-4B-GML-mix) |
| [🤗Variational-Reasoning-8B-Acc](https://huggingface.co/zhouxiangxin/Variational-Reasoning-8B-Acc) | Qwen3-8B-Base | [🤗Initial-Reasoning-8B](https://huggingface.co/zhouxiangxin/Initial-Reasoning-8B) | [🤗Variational-Posterior-PB-8B](https://huggingface.co/zhouxiangxin/Variational-Posterior-PB-8B) | [🤗Variational-Posterior-8B-Acc-mix](https://huggingface.co/datasets/zhouxiangxin/Variational-Posterior-8B-Acc-mix) |
| [🤗Variational-Reasoning-8B-GML](https://huggingface.co/zhouxiangxin/Variational-Reasoning-8B-GML) | Qwen3-8B-Base | [🤗Initial-Reasoning-8B](https://huggingface.co/zhouxiangxin/Initial-Reasoning-8B) | [🤗Variational-Posterior-PB-8B](https://huggingface.co/zhouxiangxin/Variational-Posterior-PB-8B) | [🤗Variational-Posterior-8B-GML-mix](https://huggingface.co/datasets/zhouxiangxin/Variational-Posterior-8B-GML-mix) |
| [🤗Variational-Reasoning-PA-7B-Acc](https://huggingface.co/zhouxiangxin/Variational-Reasoning-PA-7B-Acc) | Qwen2.5-7B-Instruct | [🤗Initial-Reasoning-7B](https://huggingface.co/zhouxiangxin/Initial-Reasoning-7B) | [🤗Variational-Posterior-PA-7B](https://huggingface.co/zhouxiangxin/Variational-Posterior-PA-7B) | [🤗Variational-Posterior-PA-7B-Acc-mix](https://huggingface.co/datasets/zhouxiangxin/Variational-Posterior-PA-7B-Acc-mix) |
| [🤗Variational-Reasoning-PB-7B-Acc](https://huggingface.co/zhouxiangxin/Variational-Reasoning-PB-7B-Acc) | Qwen2.5-7B-Instruct | [🤗Initial-Reasoning-7B](https://huggingface.co/zhouxiangxin/Initial-Reasoning-7B) | [🤗Variational-Posterior-PB-7B](https://huggingface.co/zhouxiangxin/Variational-Posterior-PB-7B) | [🤗Variational-Posterior-PB-7B-Acc-mix](https://huggingface.co/datasets/zhouxiangxin/Variational-Posterior-PB-7B-Acc-mix) |
| [🤗Variational-Reasoning-PA-7B-GML](https://huggingface.co/zhouxiangxin/Variational-Reasoning-PA-7B-GML) | Qwen2.5-7B-Instruct | [🤗Initial-Reasoning-7B](https://huggingface.co/zhouxiangxin/Initial-Reasoning-7B) | [🤗Variational-Posterior-PA-7B](https://huggingface.co/zhouxiangxin/Variational-Posterior-PA-7B) | [🤗Variational-Posterior-PA-7B-GML-mix](https://huggingface.co/datasets/zhouxiangxin/Variational-Posterior-PA-7B-GML-mix) |
| [🤗Variational-Reasoning-PB-7B-GML](https://huggingface.co/zhouxiangxin/Variational-Reasoning-PB-7B-GML) | Qwen2.5-7B-Instruct | [🤗Initial-Reasoning-7B](https://huggingface.co/zhouxiangxin/Initial-Reasoning-7B) | [🤗Variational-Posterior-PB-7B](https://huggingface.co/zhouxiangxin/Variational-Posterior-PB-7B) | [🤗Variational-Posterior-PB-7B-GML-mix](https://huggingface.co/datasets/zhouxiangxin/Variational-Posterior-PB-7B-GML-mix) |
| [🤗Variational-Reasoning-32B-Acc](https://huggingface.co/zhouxiangxin/Variational-Reasoning-32B-Acc) | Qwen2.5-32B-Instruct | [🤗Initial-Reasoning-32B](https://huggingface.co/zhouxiangxin/Initial-Reasoning-32B) | [🤗Variational-Posterior-PA-32B](https://huggingface.co/zhouxiangxin/Variational-Posterior-PA-32B) | [🤗Variational-Posterior-32B-Acc-mix](https://huggingface.co/datasets/zhouxiangxin/Variational-Posterior-32B-Acc-mix) |
| [🤗Variational-Reasoning-32B-GML](https://huggingface.co/zhouxiangxin/Variational-Reasoning-32B-GML) | Qwen2.5-32B-Instruct | [🤗Initial-Reasoning-32B](https://huggingface.co/zhouxiangxin/Initial-Reasoning-32B) | [🤗Variational-Posterior-PA-32B](https://huggingface.co/zhouxiangxin/Variational-Posterior-PA-32B) | [🤗Variational-Posterior-32B-GML-mix](https://huggingface.co/datasets/zhouxiangxin/Variational-Posterior-32B-GML-mix) |

## 📦 Dependencies

This work uses two separate environments for training and evaluation.  
You can install them as follows:

```bash
# Environment for training with LLaMA-Factory
conda create -n llama_factory python=3.10 -y
conda activate llama_factory
cd LLaMA-Factory
pip install -e ".[torch,metrics,deepspeed,vllm,wandb]" --no-build-isolation
conda deactivate

# Environment for evaluation and verification with SkyThought
conda create -n skythought python=3.10 -y
conda activate skythought
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
cd SkyThought
pip install -e .
cd ..
pip install timeout-decorator==0.5.0
conda deactivate
```

## 🔎 Evaluation

You can directly evaluate the provided final reasoning models $\pi_\theta$.

Please refer to `SkyThought/variational_reasoning/eval/eval.sh`.

Notes:
- SkyThought requires `vllm==0.7.0` by default. For Qwen3-based models, please upgrade to `vllm==0.8.4`.

- For 32B models, set `tensor_parallel_size=4`.


## 🚀 Training

We provide scripts to reproduce experiments.

For example, training with **Qwen3-4B-Base** is included; other experiments can be reproduced by changing the model and dataset path.

All training scripts assume 2 nodes (2 x 8 H100 GPUs). If using a different number of nodes, adjust `gradient_accumulation_steps` in `LLaMA-Factory/examples/variational_reasoning/*.yaml` accordingly to keep the effective batch size constant.

For some training steps (e.g., 1/2/9 below), you must precompute the value `constant_loss_normalizer` used in `LLaMA-Factory/examples/variational_reasoning/*.yaml`:

```bash
cd LLaMA-Factory
python -m variational_reasoning.data_process.compute_normalizer \
  --dataset_name ${dataset_name} \
  --base_model_name ${base_model_name}
cd ..
```

### 1. Train the initial reasoning model $\pi_{\theta_0}$

Please refer to `LLaMA-Factory/variational_reasoning/train/train_initial_reasoning_model`.

### 2. Train the variational posterior $q_\phi$

Please refer to `LLaMA-Factory/variational_reasoning/train/train_variational_posterior`.

### 3. Sample from the variational posterior $q_\phi$

Run **8 times** with **seeds 0–7**:
```bash
python variational_reasoning/sample_from_variational_posterior.py
  --prompt_template B \
  --model_name "zhouxiangxin/Variational-Posterior-PB-4B" \
  --seed ${seed}
```

### 4. Estimate log likelihood using the initial reasoning model $\pi_{\theta_0}$

Run **8 times** with **split_idx=0–7**: 

```bash
deepspeed --num_gpus 8 --master_port=`bash get_free_port.sh` \
  variational_reasoning/estimate_logp_initial_reasoning_model.py \
  --posterior_name "zhouxiangxin/Variational-Posterior-PB-4B" \
  --initial_reasoning_model "zhouxiangxin/Initial-Reasoning-4B" \
  --split_idx ${split_idx}
```

### 5. Estimate log likelihood using the variational posterior $q_\phi$

Run **8 times** with **split_idx=0–7**: 

```bash
deepspeed --num_gpus 8 --master_port=`bash get_free_port.sh` \
  variational_reasoning/estimate_logp_variational_posterior.py \
  --posterior_name "zhouxiangxin/Variational-Posterior-PB-4B" \
  --split_idx ${split_idx}
```

### 6. Sample from the initial reasoning model $\pi_{\theta_0}$ (Optional, required by accuracy-based estimator)

Run **8 times** with **split_idx=0–7**: 

```bash
python variational_reasoning/sample_from_initial_reasoning_model.py \
  --posterior_name "zhouxiangxin/Variational-Posterior-PB-4B" \
  --initial_reasoning_model "zhouxiangxin/Initial-Reasoning-4B" \
  --split_idx ${split_idx}
```

### 7. Verify (Optional, required by accuracy-based estimator)

Run **8 times** with **dataset_idx=0–7**: 

```bash
python -m variational_reasoning.verify.verify_parallel --dataset_idx ${dataset_idx}
```

### 8. Build the dataset for training the final reasoning model $\pi_\theta$

Choose one of the estimators:
```bash
# option: use an estimator of \pi(Y|x,z) based on geometric mean of token likelihood
python variational_reasoning/build_data_GML.py

# option: use an estimator of \pi(Y|x,z) based on accuracy
python variational_reasoning/build_data_Acc.py
```

### 9. Train the final reasoning model $\pi_\theta$

Please refer to `LLaMA-Factory/variational_reasoning/train/train_variational_posterior`.


## 📖 Citation

If you find this code useful, please consider citing our paper:
```
TBD
```
