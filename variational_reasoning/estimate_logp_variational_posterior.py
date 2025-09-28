"""
deepspeed --num_gpus 8 --master_port=`bash get_free_port.sh` variational_reasoning/data_process/estimate_logp_variational_posterior.py --split_idx 
"""
import argparse
import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import deepspeed
import numpy as np
import pickle
from huggingface_hub import login
import datasets
from tqdm.auto import tqdm
import torch.distributed as dist

deepspeed.init_distributed()

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--split_idx', type=int, required=True, help='Index of the split to process')
args_parser.add_argument('--posterior_name', type=str, default='zhouxiangxin/Variational-Posterior-PB-4B', help='specify the model path of trained variational poseterior')
args_parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from deepspeed')  # this will provided automatically by DeepSpeed, so we should not specify it manually 

args = args_parser.parse_args() 
print(f'split_idx={args.split_idx}')

if '/' in args.posterior_name:
    clean_posterior_name = args.posterior_name.split('/')[-1]
else:
    clean_posterior_name = args.posterior_name

dataset_name = f"{clean_posterior_name}-{args.split_idx}"


hf_dataset = datasets.load_dataset(f'zhouxiangxin/{dataset_name}')['train']  # users should specify their own huggingface account ID

MODEL_NAME = args.posterior_name

# config DeepSpeed Zero-3
deepspeed_config = {
    "train_batch_size": 8,  # total batch size
    "train_micro_batch_size_per_gpu": 1,  # batch size per GPU
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 3,  # use Zero-3
        "offload_optimizer": {"device": "cpu"},  
        "offload_param": {"device": "cpu"},  
        "stage3_max_live_parameters": 1e9,  
        "stage3_prefetch_bucket_size": 5e8,  
        "contiguous_gradients": True,  
        "overlap_comm": True,  
    },
    "bf16": {"enabled": True}, 
    "steps_per_print": 50,
}

class TextDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, max_seq_len=32000):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len  
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        prompt_idx = self.hf_dataset[idx]['prompt_idx']
        prompt = self.hf_dataset[idx]['prompt'] 
        response = self.hf_dataset[idx]['posterior_responses']
        
        prompt_id = tuple(self.tokenizer(prompt)['input_ids'])
        response_id = tuple(self.tokenizer(response)['input_ids'])
        

        
        prompt_len = len(prompt_id)
        response_len = len(response_id)
        
        if prompt_len + response_len > self.max_seq_len:
            print(f'Warning: Combined length of prompt and response exceeds max_seq_len ({self.max_seq_len}). Truncating.')

        prompt_len = min(prompt_len, self.max_seq_len)
        response_len = min(response_len, self.max_seq_len - prompt_len)
        
        prompt_id = prompt_id[:prompt_len]
        response_id = response_id[:response_len]
        
        input_id = list(prompt_id) + list(response_id) 
        input_id = torch.tensor(input_id)
        attention_mask = torch.ones(len(input_id))
        
        
        return {
            'prompt_idx': prompt_idx,
            'input_ids': input_id,
            'attention_mask': attention_mask,
            'prompt_len': prompt_len,
            'response_len': response_len,
        }

def zero_pad_sequences(sequences, value=0, side="right"):
    """ padding """
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []
    
    for seq in sequences:
        pad_size = max_len - len(seq)
        if side == "right":
            padded_seq = torch.cat([seq, torch.full((pad_size,), value, dtype=seq.dtype)])
        else:  # left padding
            padded_seq = torch.cat([torch.full((pad_size,), value, dtype=seq.dtype), seq])
        padded_sequences.append(padded_seq)
    
    return torch.stack(padded_sequences)

def dynamic_collate_fn(batch):
    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    pad_value = 0  # attention_mask filling value
    
    # use padding right
    input_ids = zero_pad_sequences(
        [item['input_ids'] for item in batch], 
        value=pad_token_id,
        side="right"
    )
    
    attention_mask = zero_pad_sequences(
        [item['attention_mask'] for item in batch], 
        value=pad_value,
        side="right"
    )
    
    return {
        'prompt_idx': [item['prompt_idx'] for item in batch],
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'prompt_len': torch.tensor([item['prompt_len'] for item in batch]),
        'response_len': torch.tensor([item['response_len'] for item in batch]),
    }

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# initialize DeepSpeed model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  
).eval()

model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    config=deepspeed_config
)

print(f"Rank {dist.get_rank()}: Model initialized on device {model_engine.device}")

## initialize dataset
dataset = TextDataset(hf_dataset, tokenizer)

sampler = torch.utils.data.distributed.DistributedSampler(
    dataset, 
    num_replicas=dist.get_world_size(),
    rank=dist.get_rank(),
    shuffle=False,
    drop_last=False  # avoid to drop any data
)

dataloader = DataLoader(
    dataset, 
    batch_size=deepspeed_config["train_micro_batch_size_per_gpu"],
    collate_fn=dynamic_collate_fn,
    sampler=sampler,  
    num_workers=4,
    pin_memory=True
)

@torch.no_grad()
def compute_logprobs():
    model_engine.module.eval()  
    response_logps_list = []
    
    # Each GPU only processes its own portion of data
    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        # Move data to the model's device
        input_ids = batch['input_ids'].to(model_engine.device)
        attention_mask = batch['attention_mask'].to(model_engine.device)

        # print(f'input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}')

        assert len(input_ids) == 1, "Batch size must be 1 for log probability computation"
        
        prompt_idx = batch['prompt_idx'][0]
        prompt_len = batch['prompt_len'][0].item()
        response_len = batch['response_len'][0].item()
        
        # Use DeepSpeed engine for inference
        outputs = model_engine(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # Prepare for log probability computation
        shift_logits = logits[..., :-1, :]#.contiguous()
        shift_labels = input_ids[..., 1:]#.contiguous()
        response_logits = shift_logits[..., prompt_len - 1 : prompt_len + response_len - 1, :]#.contiguous()
        response_labels = shift_labels[..., prompt_len -1 : prompt_len + response_len - 1]
        del shift_logits  
        del shift_labels
        
        # Compute log probabilities
        response_log_probs = torch.log_softmax(response_logits.float(), dim=-1)
        response_logprobs = torch.gather(
            response_log_probs, 
            dim=-1, 
            index=response_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        response_logps_list.append((prompt_idx,response_logprobs[0].cpu().numpy().tolist()))
        del response_logprobs, response_log_probs, response_logits, response_labels
        torch.cuda.empty_cache()
    
    return response_logps_list

local_logprobs = compute_logprobs()

dist.barrier()  # make sure all processes finish


import time
import random

max_retries = 3  # Maximum retry attempts
initial_wait = 1  # Initial wait time (seconds)
backoff_factor = 2  # Exponential backoff factor
jitter_range = 0.5  # Random jitter range (avoid process synchronization retries)

def gather_with_retry(local_data):
    world_size = dist.get_world_size()
    gathered_data = [None] * world_size
    attempt = 0
    success = False

    while attempt < max_retries and not success:
        try:
            # Synchronize all processes (ensure consistent state during retries)
            dist.barrier()
            
            # Perform communication operation
            dist.gather_object(
                local_data,
                gathered_data if dist.get_rank() == 0 else None,
                dst=0
            )
            success = True
            
        except RuntimeError as e:
            if "NCCL Error" not in str(e):
                raise  # Non-NCCL errors are raised directly
            
            attempt += 1
            if attempt >= max_retries:
                raise RuntimeError(f"Failed after {max_retries} retries") from e
                
            # Exponential backoff + random jitter
            wait_time = initial_wait * (backoff_factor ** (attempt - 1))
            jitter = wait_time * random.uniform(-jitter_range, jitter_range)
            actual_wait = max(0.1, wait_time + jitter)  # Avoid negative wait time
            
            if dist.get_rank() == 0:
                print(f"⚠️ NCCL error (Attempt {attempt}/{max_retries}), waiting {actual_wait:.2f} seconds before retry")
            time.sleep(actual_wait)
    
    return gathered_data

all_logprobs = gather_with_retry(local_logprobs)

        
dist.barrier()  # Ensure all processes have completed computation
        
        
# Only rank 0 saves and uploads results
if dist.get_rank() == 0:
    # Combine results from all GPUs
    flat_logprobs = [item for sublist in all_logprobs for item in sublist]
    flat_logprobs.sort(key=lambda x: x[0])  # Sort by prompt_idx
    
    sample_idx_set = set()
    new_flag_logprobs = []
    for lp in flat_logprobs:
        if lp[0] not in sample_idx_set: 
            sample_idx_set.add(lp[0])
            new_flag_logprobs.append(lp[1])
    flat_logprobs = new_flag_logprobs  
    
    # Ensure result counts match
    assert len(flat_logprobs) == len(hf_dataset), f"Response log probabilities length mismatch: len(flat_logprobs)={len(flat_logprobs)}, len(hf_dataset)={len(hf_dataset)}"
    
    # Add column and upload
    hf_dataset = hf_dataset.add_column('response_logps', flat_logprobs)
    datasetdict = datasets.DatasetDict({'train': hf_dataset})

    if '/' in dataset_name:
        dataset_name = dataset_name.split('/')[-1]
    dst_repo_name = f'zhouxiangxin/{dataset_name}-posterior_logps'
    print(f"Results uploaded to {dst_repo_name}")
    datasetdict.push_to_hub(dst_repo_name, private=True)
