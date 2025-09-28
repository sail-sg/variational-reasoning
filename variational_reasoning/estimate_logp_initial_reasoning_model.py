import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import deepspeed
from jinja2 import Template
import datasets
from tqdm.auto import tqdm
import torch.distributed as dist

def apply_bespoke_stratos_reasoning_template(question: str):
    
    prompt = Template("<|im_start|>system\nYour role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:<|im_end|>\n<|im_start|>user\n{{question}}<|im_end|>\n<|im_start|>assistant\n<|begin_of_thought|>\n\n", keep_trailing_newline=True).render(question=question)
    
    return prompt
    

deepspeed.init_distributed()

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--split_idx', type=int, required=True, help='Index of the split to process')
args_parser.add_argument('--posterior_name', type=str, default='zhouxiangxin/Variational-Posterior-PB-4B', help='specify the model path of trained variational poseterior')
args_parser.add_argument('--initial_reasoning_model', type=str, default='zhouxiangxin/Initial-Reasoning-4B', help='specify the model path of trained initial reasoning model')
args_parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from deepspeed')  # this will provided automatically by DeepSpeed, so we should not specify it manually 

args = args_parser.parse_args() 
print(f'split_idx={args.split_idx}')


if '/' in args.posterior_name:
    clean_posterior_name = args.posterior_name.split('/')[-1]
else:
    clean_posterior_name = args.posterior_name

dataset_name = f"{clean_posterior_name}-{args.split_idx}"
hf_dataset = datasets.load_dataset(f'zhouxiangxin/{dataset_name}')['train']  # users should specify their own huggingface account ID
bespoke_dataset = datasets.load_dataset('zhouxiangxin/Bespoke-Stratos-17k-Reformatted')['train']

assert len(hf_dataset) == len(bespoke_dataset)

MODEL_NAME = args.initial_reasoning_model # e.g., 'zhouxiangxin/Initial-Reasoning-4B'


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
    def __init__(self, hf_dataset, bespoke_dataset, tokenizer, max_seq_len=32000):
        self.hf_dataset = hf_dataset
        self.bespoke_dataset = bespoke_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len  
        
    def __len__(self):
        return len(self.hf_dataset)
    
    def __getitem__(self, idx):
        prompt_idx = self.hf_dataset[idx]['prompt_idx']
        prompt = apply_bespoke_stratos_reasoning_template(self.bespoke_dataset[idx]['question_content'])
        posterior_response = self.hf_dataset[idx]['posterior_responses']
        answer_text = Template("_thought|>\n\n<|begin_of_solution|>{{solution}}<|end_of_solution|>").render(solution=self.bespoke_dataset[idx]['solution_content'])    

        prompt_id = tuple(self.tokenizer(prompt)['input_ids'])
        posterior_response_id = tuple(self.tokenizer(posterior_response)['input_ids'])
        answer_id = tuple(self.tokenizer(answer_text)['input_ids'])
        
        prompt_len = len(prompt_id) 
        answer_len = len(answer_id)
        if prompt_len + answer_len + len(posterior_response_id) > self.max_seq_len:
            print(f"Warning: Prompt and answer length exceeds max_seq_len ({self.max_seq_len}). Truncating variational posterior response.")
        think_len = min(len(posterior_response_id), self.max_seq_len - prompt_len - answer_len)
        think_len = max(think_len, 1)  # make sure think_len >0
        think_id = posterior_response_id[:think_len] 
        think_text = self.tokenizer.decode(think_id, skip_special_tokens=False)        
        
        think_index = think_text.find('<|end_of')
        if think_index != -1:
            think_text = think_text[:think_index] + '<|end_of'
        else:
            think_text = think_text + '<|end_of'
        
        think_id = self.tokenizer(think_text)['input_ids']
        think_len = len(think_id)

        input_id = list(prompt_id) + list(think_id) + list(answer_id) 
        input_id = torch.tensor(input_id)
        attention_mask = torch.ones(len(input_id))
        
        return {
            'prompt_idx': prompt_idx,
            'input_ids': input_id,
            'attention_mask': attention_mask,
            'prompt_len': prompt_len,
            'think_len': think_len,
            'answer_len': answer_len,
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
        'think_len': torch.tensor([item['think_len'] for item in batch]),
        'answer_len': torch.tensor([item['answer_len'] for item in batch]),
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
dataset = TextDataset(hf_dataset, bespoke_dataset, tokenizer)

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
    think_logps_list = []
    answer_logps_list = []
    
    # each GPU process its own data
    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        input_ids = batch['input_ids'].to(model_engine.device)
        attention_mask = batch['attention_mask'].to(model_engine.device)

        assert len(input_ids) == 1, "Batch size must be 1 for log probability computation"
        
        prompt_idx = batch['prompt_idx'][0]
        prompt_len = batch['prompt_len'][0].item()
        think_len = batch['think_len'][0].item()
        answer_len = batch['answer_len'][0].item()
        
        outputs = model_engine(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        
        # compute log likelihood
        shift_logits = logits[..., :-1, :]#.contiguous()
        shift_labels = input_ids[..., 1:]#.contiguous()
        think_logits = shift_logits[..., prompt_len - 1 : prompt_len + think_len - 1, :]
        think_labels = shift_labels[..., prompt_len - 1 : prompt_len + think_len - 1]
        answer_logits = shift_logits[..., prompt_len + think_len - 1 : prompt_len + think_len + answer_len - 1, :]
        answer_labels = shift_labels[..., prompt_len + think_len - 1 : prompt_len + think_len + answer_len - 1]
        del shift_logits  
        del shift_labels
        
        think_log_probs = torch.log_softmax(think_logits.float(), dim=-1)
        think_logprobs = torch.gather(
            think_log_probs, 
            dim=-1, 
            index=think_labels.unsqueeze(-1)
        ).squeeze(-1)
        think_logps_list.append((prompt_idx, think_logprobs[0].cpu().numpy().tolist())) 
        del think_logprobs, think_log_probs, think_logits, think_labels
        
        answer_log_probs = torch.log_softmax(answer_logits.float(), dim=-1)
        answer_logprobs = torch.gather(
            answer_log_probs, 
            dim=-1, 
            index=answer_labels.unsqueeze(-1)
        ).squeeze(-1)
        answer_logps_list.append((prompt_idx, answer_logprobs[0].cpu().numpy().tolist())) 
        del answer_logprobs, answer_log_probs, answer_logits, answer_labels
        torch.cuda.empty_cache()
    
    return think_logps_list, answer_logps_list



local_think_logprobs, local_answer_logprobs = compute_logprobs()

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



all_think_logprobs = gather_with_retry(local_think_logprobs)
all_answer_logprobs = gather_with_retry(local_answer_logprobs)


        
dist.barrier()  # Ensure all processes have completed computation


def merge_and_sort_logprobs(gathered_logprobs):
    """Merge and sort by prompt_idx"""
    flat_logprobs = [item for sublist in gathered_logprobs for item in sublist]
    flat_logprobs.sort(key=lambda x: x[0])  # Sort by prompt_idx
    
    sample_idx_set = set()
    new_flag_logprobs = []
    for lp in flat_logprobs:
        if lp[0] not in sample_idx_set: 
            sample_idx_set.add(lp[0])
            new_flag_logprobs.append(lp[1])
    
    return new_flag_logprobs
        
        
# Only rank 0 saves and uploads results
if dist.get_rank() == 0:
    flat_think_logprobs = merge_and_sort_logprobs(all_think_logprobs) 
    flat_answer_logprobs = merge_and_sort_logprobs(all_answer_logprobs)
    
    # Ensure result counts match
    assert len(flat_think_logprobs) == len(hf_dataset), f"Response log probabilities length mismatch: len(flat_logprobs)={len(flat_think_logprobs)}, len(hf_dataset)={len(hf_dataset)}"
    assert len(flat_answer_logprobs) == len(hf_dataset), f"Answer log probabilities length mismatch: len(flat_answer_logprobs)={len(flat_answer_logprobs)}, len(hf_dataset)={len(hf_dataset)}" 
    
    # Add columns and upload
    hf_dataset = hf_dataset.add_column('think_logps', flat_think_logprobs)
    hf_dataset = hf_dataset.add_column('answer_logps', flat_answer_logprobs)
    datasetdict = datasets.DatasetDict({'train': hf_dataset})
    if '/' in dataset_name:
        dataset_name = dataset_name.split('/')[-1]
    dst_repo_name = f'zhouxiangxin/{dataset_name}-initial_logps'
    print(f"Results uploaded to {dst_repo_name}")
    datasetdict.push_to_hub(dst_repo_name, private=True)