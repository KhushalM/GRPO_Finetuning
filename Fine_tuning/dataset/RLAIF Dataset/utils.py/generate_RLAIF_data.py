"""
Fixed dataset generation script that creates proper JSON arrays with messages format for fine-tuning
"""

import argparse
import json
import os
import openai
import random
import sys
import time
from datetime import datetime
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load environment variables from .env file
load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset for first principles")
    parser.add_argument("--base_model", type=str, default="gpt-4-turbo", help="Base SFT model to use for generation")
    parser.add_argument("--preference_model", type=str, default="gpt-4o-mini", help="Preference model to use for generation")
    parser.add_argument("--num_generations", type=int, default=600, help="Number of generations to generate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generation")
    parser.add_argument("--output_file", type=str, default="preference_dataset.json", help="File to save the dataset")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    return parser.parse_args()

def load_dataset(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

file_path = "/Users/khushal/Documents/GitHub/GRPO_Finetuning/Fine_tuning/dataset/structured_dataset.json"

def gather_system_and_user_prompts(file_path):
    system_prompts = []
    user_prompts = []
    dataset = load_dataset(file_path)
    system_prompts = [dataset[0]["messages"][0]["content"]]

    for item in dataset:
        user_prompts.append(item["messages"][1]["content"])

    return system_prompts, user_prompts

def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def generate_2_responses_batch(model_name, file_path, batch_size, num_generations):
    """Generate responses in batches for efficiency"""
    completions = []
    tokenizer, model = load_model("merged_model")
    
    system_prompts, user_prompts = gather_system_and_user_prompts(file_path)
    system_prompt = system_prompts[0]
    
    # Limit to num_generations or available prompts
    prompts_to_process = user_prompts[:min(num_generations, len(user_prompts))]
    
    print(f"Processing {len(prompts_to_process)} prompts in batches of {batch_size}")
    
    # Process prompts in batches
    for i in range(0, len(prompts_to_process), batch_size):
        batch_prompts = prompts_to_process[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(len(prompts_to_process) - 1) // batch_size + 1}")
        
        batch_completions = []
        
        for prompt in batch_prompts:
            # Prepare the conversation format
            combined_prompt = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            try:
                # Tokenize the prompt
                combined_prompt_tokenized = tokenizer.apply_chat_template(
                    combined_prompt, 
                    return_tensors="pt", 
                    add_generation_prompt=True
                )
                
                # Generate 2 responses for each prompt
                outputs = model.generate(
                    combined_prompt_tokenized, 
                    max_length=512, 
                    temperature=0.7, 
                    do_sample=True, 
                    num_return_sequences=2, 
                    pad_token_id=tokenizer.eos_token_id
                )
                
                # Decode responses
                response_a = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response_b = tokenizer.decode(outputs[1], skip_special_tokens=True)
                
                # Clean responses (remove input prompt from output)
                input_text = tokenizer.decode(combined_prompt_tokenized[0], skip_special_tokens=True)
                response_a = response_a.replace(input_text, "").strip()
                response_b = response_b.replace(input_text, "").strip()
                
                completion = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "Response A": response_a,
                    "Response B": response_b,
                }
                
                batch_completions.append(completion)
                
            except Exception as e:
                print(f"Error processing prompt: {e}")
                continue
        
        completions.extend(batch_completions)
        
        # Add small delay between batches to prevent overloading
        if i + batch_size < len(prompts_to_process):
            time.sleep(1)
    
    return completions

def ai_preference_dataset_batch(preference_model_name, dataset, batch_size):
    """Process preference evaluation in batches"""
    preference_dataset = []
    
    system_prompt = """You are an expert evaluator of first-principles explanations. 
Which answer from Response A and Response B is better for clarity, correctness, and depth in terms of explaining the concept like Richard Feynman? 
Answer exactly A or B."""
    
    print(f"Processing {len(dataset)} preference evaluations in batches of {batch_size}")
    
    # Process dataset in batches
    for i in range(0, len(dataset), batch_size):
        batch_data = dataset[i:i + batch_size]
        print(f"Processing preference batch {i // batch_size + 1}/{(len(dataset) - 1) // batch_size + 1}")
        
        batch_preferences = []
        
        for item in batch_data:
            main_system_prompt = item["messages"][0]["content"]
            prompt = item["messages"][1]["content"]
            response_a = item["Response A"]
            response_b = item["Response B"]
            
            user_prompt = f"""
Question: {prompt}
Response A: {response_a}
Response B: {response_b}
Answer with A or B only.
"""
            
            try:
                response = client.chat.completions.create(
                    model=preference_model_name,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        }
                    ],
                    temperature=0.0,
                    max_tokens=1,
                )
                
                preference = response.choices[0].message.content.strip()
                
                preference_item = {
                    "messages": [
                        {
                            "role": "system",
                            "content": main_system_prompt
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "Response A": response_a,
                    "Response B": response_b,
                    "Preference": preference
                }
                
                batch_preferences.append(preference_item)
                
            except Exception as e:
                print(f"Error processing preference evaluation: {e}")
                # Add delay and retry once
                time.sleep(2)
                try:
                    response = client.chat.completions.create(
                        model=preference_model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            }
                        ],
                        temperature=0.0,
                        max_tokens=1,
                    )
                    
                    preference = response.choices[0].message.content.strip()
                    
                    preference_item = {
                        "messages": [
                            {
                                "role": "system",
                                "content": main_system_prompt
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "Response A": response_a,
                        "Response B": response_b,
                        "Preference": preference
                    }
                    
                    batch_preferences.append(preference_item)
                    
                except Exception as retry_error:
                    print(f"Retry failed: {retry_error}")
                    continue
        
        preference_dataset.extend(batch_preferences)
        
        # Add delay between batches to respect API rate limits
        if i + batch_size < len(dataset):
            time.sleep(2)
    
    return preference_dataset

def convert_to_json(dataset, output_file):
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)

def main():
    args = parse_args()
    
    print(f"Starting dataset generation with batch size: {args.batch_size}")
    print(f"Target generations: {args.num_generations}")
    
    # Generate responses in batches
    completions = generate_2_responses_batch(
        args.base_model, 
        file_path, 
        args.batch_size, 
        args.num_generations
    )
    
    print(f"Generated {len(completions)} response pairs")
    
    # Generate preference dataset in batches
    preference_dataset = ai_preference_dataset_batch(
        args.preference_model, 
        completions, 
        args.batch_size
    )
    
    print(f"Generated {len(preference_dataset)} preference evaluations")
    
    # Save the dataset
    convert_to_json(preference_dataset, args.output_file)
    print(f"Dataset saved to {args.output_file}")

if __name__ == "__main__":
    main()
    