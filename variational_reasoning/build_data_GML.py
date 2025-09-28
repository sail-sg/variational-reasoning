import argparse
import os
import torch
from jinja2 import Template
import datasets
from tqdm.auto import tqdm

def apply_bespoke_stratos_reasoning_prompt_template(question: str):
    
    prompt = Template("<|im_start|>system\nYour role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {thought with steps separated with '\\n\\n'} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {final formatted, precise, and clear solution} <|end_of_solution|> Now, try to solve the following question through the above guidelines:<|im_end|>\n<|im_start|>user\n{{question}}<|im_end|>\n<|im_start|>assistant\n<|begin_of_thought|>\n\n", keep_trailing_newline=True).render(question=question)
    
    return prompt

def apply_bespoke_stratos_reasoning_response_template(think_content: str, solution_content: str):
    response = Template("{{think_content}}<|end_of_thought|>\n\n<|begin_of_solution|>{{solution_content}}<|end_of_solution|>", keep_trailing_newline=True).render(think_content=think_content, solution_content=solution_content)
    return response 
    

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--posterior_name', type=str, default='zhouxiangxin/Variational-Posterior-PB-4B', help='specify the model path of trained variational poseterior')
args = args_parser.parse_args() 

clean_posterior_name = args.posterior_name.split('/')[-1]


bespoke_dataset = datasets.load_dataset('zhouxiangxin/Bespoke-Stratos-17k-Reformatted')['train']

dataset_list = []
posterior_logp_dataset_list = []
initial_logp_dataset_list = []


for split_idx in range(8):
    dataset_name = f"{clean_posterior_name}-{split_idx}"
    hf_dataset = datasets.load_dataset(f'zhouxiangxin/{dataset_name}')['train']
    dataset_list.append(hf_dataset)
    hf_dataset = datasets.load_dataset(f'zhouxiangxin/{dataset_name}-posterior_logps')['train']
    posterior_logp_dataset_list.append(hf_dataset)
    hf_dataset = datasets.load_dataset(f'zhouxiangxin/{dataset_name}-initial_logps')['train']
    initial_logp_dataset_list.append(hf_dataset)
    
    
res_dict = {
    'prompt': [],
    'response': [],
    'source': [],  
}

for sample_idx in tqdm(range(len(bespoke_dataset))):
    
    candidate_dataset_indices = []
    
    for dataset_idx in range(len(dataset_list)):
        
        sample = dataset_list[dataset_idx][sample_idx]
        posterior_logps = posterior_logp_dataset_list[dataset_idx][sample_idx]['response_logps']
        initial_think_logps = initial_logp_dataset_list[dataset_idx][sample_idx]['think_logps']
        initial_answer_logps = torch.tensor(initial_logp_dataset_list[dataset_idx][sample_idx]['answer_logps'])
        

        shared_think_len = min(len(posterior_logps), len(initial_think_logps)) 
        posterior_logps = torch.tensor(posterior_logps[:shared_think_len])
        initial_think_logps = torch.tensor(initial_think_logps[:shared_think_len])
        
        
        think_logp_diff = (initial_think_logps - posterior_logps).mean()
        
        if shared_think_len < 28000: ## drop too long responses
            # the frist term is arithmetic mean of \log \pi_\theta(Y | x, z), which is equivalent to log of geometric mean of \pi_\theta(Y | x, z)
            # the second term is geometric mean of log ratio, which is equivalent to log of geometric mean of likelihood ratio
            candidate_dataset_indices.append((initial_answer_logps.mean().item() + think_logp_diff.item(), dataset_idx))

    if len(candidate_dataset_indices) > 0:
        candidate_dataset_indices.sort(key=lambda x: x[0], reverse=True)
        best_dataset_idx = candidate_dataset_indices[0][1]
        
        sample = dataset_list[best_dataset_idx][sample_idx]
        posterior_response = sample['posterior_responses']
        
        think_index = posterior_response.find('<|end_of')
        if think_index != -1:
            think_text = posterior_response[:think_index] + '<|end_of'
        else:
            think_text = posterior_response + '<|end_of'
            
        answer_text = Template("_thought|>\n\n<|begin_of_solution|>{{solution}}<|end_of_solution|>").render(solution=bespoke_dataset[sample_idx]['solution_content'])    
        response = think_text + answer_text
        res_dict['source'].append(best_dataset_idx)
    else:
        response = apply_bespoke_stratos_reasoning_response_template(bespoke_dataset[sample_idx]['think_content'], bespoke_dataset[sample_idx]['solution_content'])
        res_dict['source'].append(-1)

    prompt = apply_bespoke_stratos_reasoning_prompt_template(bespoke_dataset[sample_idx]['question_content'])
    res_dict['prompt'].append(prompt)
    res_dict['response'].append(response)

dataset = datasets.Dataset.from_dict(res_dict)
datadict = datasets.DatasetDict({'train': dataset})

# dst_repo_name = f'zhouxiangxin/{clean_posterior_name}-GML-posterior'
# print(f"Results uploaded to {dst_repo_name}")
# datadict.push_to_hub(dst_repo_name, private=True)

res_dict = {
    'prompt': [],
    'response': [],
    'source': [],  
}

for data in tqdm(bespoke_dataset):
    prompt = apply_bespoke_stratos_reasoning_prompt_template(data['question_content'])
    think_content = data['think_content']
    solution_content = data['solution_content']
    response = apply_bespoke_stratos_reasoning_response_template(think_content, solution_content)
    res_dict['prompt'].append(prompt)
    res_dict['response'].append(response)
    res_dict['source'].append(-2)

for data in tqdm(dataset):
    res_dict['prompt'].append(data['prompt'])
    res_dict['response'].append(data['response'])  
    res_dict['source'].append(data['source'])


dataset = datasets.Dataset.from_dict(res_dict)
print(dataset)
datadict = datasets.DatasetDict({'train': dataset})
dst_repo_name = f'zhouxiangxin/{clean_posterior_name}-GML-mix'
print(f"Results uploaded to {dst_repo_name}")
datadict.push_to_hub(dst_repo_name, private=True)



