import argparse

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--posterior_name', type=str, default='zhouxiangxin/Variational-Posterior-PB-4B', help='specify the model path of trained variational poseterior')
args_parser.add_argument('--initial_reasoning_model', type=str, default="zhouxiangxin/Initial-Reasoning-4B", required=True, help='Model path of initial reasoning model')
args_parser.add_argument('--split_idx', type=int, required=True, help='Index of the split to process')
args = args_parser.parse_args() 
print(f'split_idx={args.split_idx}')


import datasets
import vllm
from jinja2 import Template
from tqdm.auto import tqdm



def apply_bespoke_stratos_reasoning_prompt_template(question: str):
    
    prompt = Template("<|im_start|>system\nYour role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:<|im_end|>\n<|im_start|>user\n{{question}}<|im_end|>\n<|im_start|>assistant\n<|begin_of_thought|>\n\n", keep_trailing_newline=True).render(question=question)
    
    return prompt
    
if '/' in args.posterior_name:
    clean_posterior_name = args.posterior_name.split('/')[-1]
else:
    clean_posterior_name = args.posterior_name

dataset_name = f"{clean_posterior_name}-{args.split_idx}"


posterior_dataset = datasets.load_dataset(f'zhouxiangxin/{dataset_name}')['train']
bespoke_dataset = datasets.load_dataset('zhouxiangxin/Bespoke-Stratos-17k-Reformatted')['train']

assert len(posterior_dataset) == len(bespoke_dataset)

model_name = args.initial_reasoning_model

vllm_args = {
    "model": model_name,
    "trust_remote_code": True,
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.9,
    "dtype": "bfloat16",
    "enable_prefix_caching": True,
    "enable_sleep_mode": False,
    "max_model_len": None,
}
llm = vllm.LLM(**vllm_args)
tokenizer = llm.get_tokenizer()

sampling_n = 8
prompt_idx_list = []
prompts = []
correct_format_list = []
for sample_idx in tqdm(range(len(bespoke_dataset)), desc='prepare prompts'):
    prompt = apply_bespoke_stratos_reasoning_prompt_template(bespoke_dataset[sample_idx]['question_content'])
    
    posterior_response = posterior_dataset[sample_idx]['posterior_responses']
    think_index = posterior_response.find('<|end_of')
    if think_index != -1:  # correct format
        think_content = posterior_response[:think_index] +'<|end_of'
        for _ in range(sampling_n):
            correct_format_list.append(1)
    else:
        think_content = posterior_response + '<|end_of'
        for _ in range(sampling_n):
            correct_format_list.append(0)
    
    for _ in range(sampling_n):
        prompts.append(prompt + think_content)
        prompt_idx_list.append(posterior_dataset[sample_idx]['prompt_idx'])


sampling_params = vllm.SamplingParams(
    temperature=0.7,
    top_p=1.0,
    top_k=-1,
    max_tokens=32764,
    include_stop_str_in_output=True,
    seed=args.split_idx,
    stop=['<|end_of_solution|>']
)

outputs = llm.generate(
    prompts, sampling_params=sampling_params, use_tqdm=True
)

model_responses = []
for i in range(len(outputs)):
    response = outputs[i].outputs[0].text 
    model_responses.append(response)


res_dict = {
    'prompt_idx': prompt_idx_list,
    'prompts': prompts,
    'correct_format': correct_format_list,
    'initial_responses': model_responses
}


dataset = datasets.Dataset.from_dict(res_dict)
datadict = datasets.DatasetDict({'train': dataset})
datadict.push_to_hub(f"zhouxiangxin/{dataset_name}-initial_response", private=True)