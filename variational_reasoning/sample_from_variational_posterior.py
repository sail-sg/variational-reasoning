import datasets
import vllm
from tqdm.auto import tqdm
import argparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, required=True, help='random seed, also use to specify the ID of generated data')
parser.add_argument('--prompt_template', type=str, choices=['A', 'B'], default='B', help='specify the prompt template used by variational posterior, A or B')
parser.add_argument('--posterior_name', type=str, default='zhouxiangxin/Variational-Posterior-PB-4B', help='specify the model path of trained variational poseterior')
args = parser.parse_args()


## load data
if args.prompt_template == 'A':
    dataset = datasets.load_dataset('zhouxiangxin/Bespoke-Stratos-17k-Train-Posterior-PA') 
else:
    dataset = datasets.load_dataset('zhouxiangxin/Bespoke-Stratos-17k-Train-Posterior-PB') 
dataset = dataset['train']
assert len(dataset) == 16710


## load trained variational posterior 
posterior_name = args.posterior_name


vllm_args = {
    "model": posterior_name,
    "trust_remote_code": True,
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.90,
    "dtype": "bfloat16",
    "enable_prefix_caching": True,
    "enable_sleep_mode": False,
    "max_model_len": None,
}
llm = vllm.LLM(**vllm_args)
tokenizer = llm.get_tokenizer()


prompts = []

for data in tqdm(dataset, desc='prepare prompts'):
    prompt = data['prompt']
    prompts.append(prompt)


sampling_params = vllm.SamplingParams(
    temperature=0.7,
    top_p=1.0,
    top_k=-1,
    max_tokens=32764,
    seed=args.seed,
)
## generate responses 
outputs = llm.generate(
    prompts, sampling_params=sampling_params, use_tqdm=True
)


model_responses = []
for i in range(len(outputs)):
    response = outputs[i].outputs[0].text 
    model_responses.append(response)

dataset = dataset.add_column('variational_posterior_responses', model_responses)
dataset = dataset.add_column("prompt_idx", list(range(len(dataset))))


datadict = datasets.DatasetDict({'train': dataset})
posterior_name = posterior_name.split('/')[-1]
dst_repo_name = f"zhouxiangxin/{posterior_name}-{args.seed}"
print(f'push to hub: {dst_repo_name}')
exit()
datadict.push_to_hub(dst_repo_name, private=True)