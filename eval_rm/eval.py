import argparse
import json

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Sample responses using vLLM.")
    parser.add_argument("--model", type=str, default='../saves/qwen2.5-7b/full/sft-rm',
                        help="The model name or path (e.g., 'meta-llama/Llama-2-7b-chat-hf').")
    parser.add_argument("--prompt", type=str, default='HanningZhang/MLE-Reward-Rating',
                        help="Prompt text or path to a .txt file containing the prompt.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature.")
    parser.add_argument("--max_tokens", type=int, default=8192,
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of completions to sample.")
    parser.add_argument("--seed",type=int, default=42)
    parser.add_argument("--tensor_parallel_size",type=int,default=1)
    parser.add_argument("--output_dir",type=str,default="gen/qwen25_output.json")
    return parser.parse_args()

def load_prompt(prompt_arg):
    # If a file path is passed, load content from file
    ds = load_dataset(prompt_arg,split='train')
    prompt_list = []
    gt_list = []
    for sample in ds:
        prompt_list.append(sample['conversations'][:1])
        gt_list.append(sample['conversations'][1:2])
    return prompt_list, gt_list

def main():
    args = parse_args()
    prompt_text,gt_list = load_prompt(args.prompt)
    
    prompt_text = prompt_text[:]
    gt_list = gt_list[:]
    print(len(prompt_text))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Initialize vLLM
    llm = LLM(model=args.model,tensor_parallel_size=args.tensor_parallel_size, dtype="bfloat16", gpu_memory_utilization=0.8, seed=args.seed)

    # Define sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        n=args.num_samples
    )
    input_list = []
    for sample in prompt_text:
        input_text = tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)
        input_list.append(sample)
    # Run generation
    print(f"Generating {args.num_samples} response(s)...\n")
    outputs = llm.generate(input_text, sampling_params)

    response_list = []
    for i, output in enumerate(outputs):
        print(f"=== Response {i+1} ===")
        print(output.outputs[0].text.strip())
        response_text = output.outputs[0].text.strip()
        response_list.append({"prompt":prompt_text[i][0]['content'],"response":response_text, "gt": gt_list[i][0]['content']})
     
    with open(args.output_dir,'w') as f:
        json.dump(response_list,f,indent=4,ensure_ascii=False)

if __name__ == "__main__":
    main()
