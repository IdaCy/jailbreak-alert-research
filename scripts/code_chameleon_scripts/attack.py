import pandas as pd
import os
import argparse
from utils import save_generation, complete_format, save_prompts
from template import get_prompts


WAIT_TIME = 10

def parse_args():
    parser = argparse.ArgumentParser(description='open_source_inference')

    # Path
    parser.add_argument('--model_path', type=str, default='', help='The path or name of the model to evaluate')
    parser.add_argument('--problem_path', type=str, default='', help='the path of the harmful problems')
    parser.add_argument('--save_path', type=str, default='', help='the path to save results')

    # Convert 
    parser.add_argument('--encrypt_rule', type=str, choices=['none', 'binary_tree', 'reverse','odd_even','length'], help='different encrypt methods')
    parser.add_argument('--prompt_style', type=str, choices=['text', 'code'], help='the style of prompt')
    parser.add_argument('--save_prompts', action='store_true', help='Whether to save prompts')


    # Model generate parameter
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--do_sample', action='store_true', help='use reward clip')
    parser.add_argument('--temperature', type=float, default=1.0, help='the max length of model generation')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='the max length of model generation')
    parser.add_argument('--top_p', type=float, default=0.9, help='the max length of model generation')
    parser.add_argument('--use_cache', action='store_true', help='use reward clip')

    args = parser.parse_args()

    return args

def set_config(model_generation_config, args):
    generation_config = model_generation_config

    generation_config.max_new_tokens = args.max_new_tokens
    generation_config.do_sample = args.do_sample
    generation_config.repetition_penalty = args.repetition_penalty
    generation_config.temperature = args.temperature
    generation_config.top_p = args.top_p
    generation_config.use_cache = args.use_cache

    return generation_config

#This converts our promts to their required format in a new csv file
def save_prompts_as_csv(args):
    """Encrypt, format, and save prompts in CSV format for inference"""
    # Get and format prompts
    prompts, original_queries = get_prompts(args)
    if prompts is None:
        raise ValueError("get_prompts returned None")
    if original_queries is None:
        raise ValueError("get_prompts returned None for original_queries")
    
    print(f"Prompts: {prompts[:5]}")  # Print first 5 prompts for debugging
    print(f"Original Queries: {original_queries[:5]}")  # Print first 5 original queries for debugging
    
    complete_prompts = complete_format(args, prompts)
    if complete_prompts is None:
        raise ValueError("complete_format returned None")
    
    print(f"Complete Prompts: {complete_prompts[:5]}")  # Print first 5 complete prompts for debugging
    
    # Convert prompts to format expected by csv_inference
    formatted_prompts = [{'sentence': prompt} for prompt in complete_prompts]
    df = pd.DataFrame(formatted_prompts)
    
    #this sets the save path
    csv_path = "../data/code_chameleon/formatted_prompts.csv"
    
    # Ensure directory exists
    os.makedirs("data/code_chameleon", exist_ok=True)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"Saved formatted prompts to: {csv_path}")
    return csv_path

def query_function(temperature, top_p, api_key, chat_prompts, args):
    client = OpenAI(api_key=api_key)
    results = []
    index = 0

    with tqdm(total=len(chat_prompts)) as pbar:
        for chat_prompt in chat_prompts:
            index = index + 1
            chat_completion = client.chat.completions.create(
                model=args.model_name,
                messages=chat_prompt,
                temperature=temperature,
                top_p=top_p
            )
            pbar.update(1)
            response = chat_completion.choices[0].message.content
            results.append(response)
            if index % 20 ==0:
                save_generation(args, results, index)
            time.sleep(WAIT_TIME) 

    return results, index

def open_source_attack(args):
    prompts, original_queries = get_prompts(args)
    complete_prompts = complete_format(args, prompts)
    
    if args.save_prompts:
        csv_path = save_prompts_as_csv(complete_prompts)
        print(f"Saved CSV format prompts to: {csv_path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,device_map="auto")

    model_generation_config = model.generation_config
    model_generation_config = set_config(model_generation_config, args)

    results = []
    index = 0
    for iters in tqdm(range(len(complete_prompts))):
        index += 1

        input_ids = tokenizer(complete_prompts[iters], return_tensors="pt").to('cuda')
        output = model.generate(**input_ids, generation_config=model_generation_config)
        prompt_len = input_ids['attention_mask'].shape[-1]
        result = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        results.append(result)

        if index % 50 == 0:
            save_generation(args,results, index)
    save_generation(args,results, index)    


def gpt_attack(args):
    prompts, original_queries = get_prompts(args)
    complete_prompts = complete_format(args, prompts)
    
    if args.save_prompts:
        csv_path = save_prompts_as_csv(complete_prompts)
    
    results, index = query_function(args.temperature, args.top_p, OPENAI_API_KEY, complete_prompts, args)
    save_generation(args, results, index)


def main(args):
    if args.model_name=='gpt':
        gpt_attack(args)
    else:
        open_source_attack(args)


if __name__ == "__main__":
    args = parse_args()
    if 'Llama' in args.model_path:
        args.model_name = 'llama2'
    elif 'vicuna' in args.model_path:
        args.model_name = 'vicuna'
    elif 'gpt' in args.model_path:
        args.model_name = 'gpt'

    if '7b' in args.model_path:
        args.model_size = '7B'
    elif '13b' in args.model_path:
        args.model_size = '13B'
    elif '70b' in args.model_path:
        args.model_size = '70B'
    elif 'gpt-4' in args.model_path:
        args.model_size = '4'
    elif 'gpt-3.5' in args.model_path:
        args.model_size = '3.5'
    main(args)