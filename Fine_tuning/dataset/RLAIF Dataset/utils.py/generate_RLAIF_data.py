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
from vllm import LLM, SamplingParams
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from datasets import Dataset
from huggingface_hub import login
import time, random 
from openai import RateLimitError, OpenAIError
import logging


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
HF_TOKEN = os.getenv("HF_TOKEN")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting dataset generation")

def safe_chat_completion(**kwargs):
    max_retries = 5
    for i in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            backoff = (2 ** i) + random.random()
            logger.warning(f"Rate limit, sleeping {backoff:.1f}s before retry")
            time.sleep(backoff)
    # final attempt (let exception bubble)
    return client.chat.completions.create(**kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset for first principles")
    parser.add_argument("--base_model", type=str, default="KhushalM/Qwen2.5-1.5-SFT-Merged",
                        help="Base SFT model to use for generation")
    parser.add_argument("--preference_model", type=str, default="gpt-4o-mini",
                        help="Preference model to use for generation")
    parser.add_argument("--num_generations", type=int,
                        default=2, help="Number of generations to generate")
    parser.add_argument("--batch_size", type=int, default=40,
                        help="Batch size for generation")
    parser.add_argument("--output_file", type=str,
                        default="preference_dataset.json", help="File to save the dataset")
    parser.add_argument("--temperature", type=float,
                        default=0.7, help="Temperature for generation")
    parser.add_argument("--device", type=str,
                        default="cuda", help="Device to use for generation")
    return parser.parse_args()


def load_dataset(file_path):
    with open(file_path, "r") as f:
        logger.info(f"Loading dataset from {file_path}")
        return json.load(f)


file_path = "structured_dataset.json"


def gather_system_and_user_prompts(file_path):
    system_prompts = []
    user_prompts = []
    dataset = load_dataset(file_path)
    system_prompts = [dataset[0]["messages"][0]["content"]]

    for item in dataset:
        user_prompts.append(item["messages"][1]["content"])
    logger.info(
        f"Loaded {len(system_prompts)} system prompts and {len(user_prompts)} user prompts")
    return system_prompts, user_prompts


def load_model(model_name, device="cuda"):
    import torch

    # Check if CUDA is available when device is set to cuda
    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Move model to specified device
    model = model.to(device)

    tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Loaded model on {device}")
    logger.info(f"Tokenizer configured")
    return tokenizer, model

def generate_2_responses_batch(model_name, file_path, batch_size, num_generations, device="cuda"):
    """Generate responses in batches for efficiency"""
    completions = []
    llm = LLM(
        model = model_name,
        dtype = "auto",
        device = device,
        tensor_parallel_size = 1,
    )
    #tokenizer, _ = load_model(model_name, device)

    system_prompts, user_prompts = gather_system_and_user_prompts(file_path)
    # system_prompt = system_prompts[0]
    # system_prompt = "You are a brilliant teacher who explains concepts like Richard Feynman.\n"


    # Process prompts in batches
    for i in range(0, len(user_prompts), batch_size):
        batch_prompts = user_prompts[i:i + batch_size]
        texts = []
        system_prompts = []
        seed = random.randint(0, 10000)
        top_p = random.choice([0.5, 0.6, 0.7, 0.8, 0.9])
        top_k = random.choice([10, 20, 30, 40, 50])
        temperature = random.uniform(0.65, 0.85)

        intros = [
            "Suppose...",
            "Picture this...",
            "Let’s consider...",
            "Imagine for a moment...",
            "Think about..."
        ]
        last_intro = None
        for prompt in batch_prompts:
            available_intros = [intro for intro in intros if intro != last_intro]
            chosen_intro = random.choice(available_intros)
            last_intro = chosen_intro
            
            system_prompt = """You are a brilliant teacher who explains concepts like Richard Feynman.

        Instructions:
        - Provide a complete, self-contained explanation
        - Never ask follow-up questions. 
        - Use relatable analogies or examples.
        - Assume no specialized background knowledge.
        - Make abstract concepts concrete and visual.
        - Write in a single focused paragraph (100-150 words maximum)
        - Be conversational but authoritative
        - End with a simple question at the end asking the user understood the concept or not.
        - Start with a random intro from the following list: {intros}"""
            
            # Create proper chat format
            chat_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Format for the model (adjust based on your model's chat template)
            formatted_text = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{chosen_intro} "
            
            system_prompts.append(system_prompt)
            texts.append(formatted_text)
        print(
            f"Processing batch {i // batch_size + 1}/{(len(user_prompts) - 1) // batch_size + 1}")

        batch_completions = []
        logger.info(
            f"Processing batch {i // batch_size + 1}/{(len(user_prompts) - 1) // batch_size + 1}")
        sampling_params = SamplingParams(
            n = 2,
            max_tokens = 256,
            temperature = 0.7,
            top_p = 0.95,
            top_k = 40,
            repetition_penalty = 1.15,
            seed = random.randint(0, 10000),
            stop=["<|im_end|>", "\n\nUser:", "\n\nHuman:", "?", "#"]
        )

        results = llm.generate(
            texts,
            sampling_params = sampling_params,
        )

        all_outputs = []
        for batch_recs in results:
            for gen in batch_recs.outputs:
                all_outputs.append(gen.text)

        for idx, (system_prompt, prompt) in enumerate(zip(system_prompts, batch_prompts)):
            resp = all_outputs[idx * 2:(idx+1)*2]
            batch_completions.append({
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
                "Response A": resp[0],
                "Response B": resp[1]
            })
        completions.extend(batch_completions)
        logger.info(
            f"Completed batch {i // batch_size + 1}/{(len(user_prompts) - 1) // batch_size + 1}")
        # Add small delay between batches to prevent overloading
        if i + batch_size < len(user_prompts):
            time.sleep(1)

    return completions


def ai_preference_dataset_batch(preference_model_name, dataset, batch_size, parallel_evaluations = 4):
    """Process preference evaluation in batches"""
    preference_dataset = []

    system_prompt = """You are an expert evaluator of educational explanations, specializing in the Feynman teaching method.

TASK: Compare two explanations and select the one that better demonstrates first-principles thinking and pedagogical effectiveness.

SCORING FRAMEWORK:
Rate each response on these criteria (but don't show scores, just use them internally):

CLARITY (30 points):
- Uses simple, concrete language over jargon
- Explanation flows logically from basic to complex
- Each sentence builds clearly on the previous

DEPTH (25 points):
- Explains WHY something happens, not just WHAT happens
- Traces back to fundamental physical/logical principles
- Avoids superficial or circular explanations

ACCESSIBILITY (25 points):
- Uses relatable analogies or examples
- Assumes no specialized background knowledge
- Makes abstract concepts concrete and visual

COMPLETENESS (20 points):
- Self-contained explanation with no critical gaps
- Addresses the core question directly
- Provides sufficient detail for understanding

RED FLAGS (automatic disqualification):
- Contains factual errors
- Uses unexplained technical terms
- Circular reasoning or hand-waving
- Asks follow-up questions instead of explaining

DECISION PROCESS:
1. Identify which response better embodies Feynman's principle: "Explain it so a child could understand"
2. Consider which would actually help someone understand the concept, not just recognize it
3. Choose the response that builds genuine insight from first principles

OUTPUT: State your choice as a single letter: "A" or "B"."""
    print(
        f"Processing {len(dataset)} preference evaluations in batches of {batch_size}")

    # Process dataset in batches
    for i in range(0, len(dataset), batch_size):
        batch_data = dataset[i:i + batch_size]
        print(
            f"Processing preference batch {i // batch_size + 1}/{(len(dataset) - 1) // batch_size + 1}")
        logger.info(
            f"Processing preference batch {i // batch_size + 1}/{(len(dataset) - 1) // batch_size + 1}")
        batch_preferences = []
        
        def evaluate_item(item):
            prompt_text = f"""
Question: {item['messages'][1]['content']}
Response A: {item['Response A']}
Response B: {item['Response B']}
Answer with A or B only.
"""
            try:
                resp = safe_chat_completion(
                    model=preference_model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": prompt_text},
                    ],
                    temperature=0.0,
                    max_tokens=1,
                    timeout=10,
                )
                choice = resp.choices[0].message.content.strip() or "A"
            except Exception as e:
                logger.error("Error for prompt '%s': %s", item['messages'][1]['content'], e)
                choice = "A"

            return {
                "messages": item["messages"],
                "Response A": item["Response A"],
                "Response B": item["Response B"],
                "Preference": choice,
            }

        with ThreadPoolExecutor(max_workers=parallel_evaluations) as executor:
            futures = {executor.submit(evaluate_item, item): item for item in batch_data}
            for future in as_completed(futures):
                batch_preferences.append(future.result())

        preference_dataset.extend(batch_preferences)

        # Add delay between batches to respect API rate limits
        if i + batch_size < len(dataset):
            time.sleep(5)

    return preference_dataset


def convert_to_json(dataset, output_file):
    logger.info(f"Converting dataset to JSON")
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=4)
    return output_file


def flatten_messages(messages):
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def convert_to_reward_format(preference_item):
    prompt = flatten_messages(preference_item["messages"])
    chosen = preference_item["Response A"] if preference_item["Preference"] == "A" else preference_item["Response B"]
    rejected = preference_item["Response A"] if preference_item["Preference"] == "B" else preference_item["Response B"]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def push_to_hub(dataset, output_file):
    logger.info(f"Pushing dataset to Hugging Face")
    converted_dataset_1 = [convert_to_reward_format(item) for item in dataset]
    output_file = convert_to_json(converted_dataset_1, output_file)
    output_file_2 = convert_to_json(dataset, "preference_dataset_old_format.json")
    # Load the JSON file back as a list of dictionaries
    with open(output_file, 'r') as f:
        dataset_list = json.load(f)
    dataset = Dataset.from_list(dataset_list)
    login(HF_TOKEN)
    #Ask before pushing to hub
    answer = input("Are you sure you want to push the dataset to Hugging Face? (y/n): ")
    if answer.lower() != "y":
        logger.info("Dataset not pushed to Hugging Face")
        return
    else:
        logger.info("Dataset pushed to Hugging Face")
    dataset.push_to_hub(
        repo_id="KhushalM/Qwen2.5-rlaif-Feynman-Dataset",
        private=False,
    )
    logger.info(f"Dataset pushed to Hugging Face")


def main():
    args = parse_args()

    print(f"Starting dataset generation with batch size: {args.batch_size}")
    print(f"Target generations: {args.num_generations}")
    print(f"Using device: {args.device}")

    # Check CUDA availability
    import torch
    if args.device == "cuda":
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA requested but not available, falling back to CPU")
            args.device = "cpu"

    # Generate responses in batches
    completions = generate_2_responses_batch(
        args.base_model,
        file_path,
        args.batch_size,
        args.num_generations,
        args.device
    )

    # Generate preference dataset in batches
    preference_dataset = ai_preference_dataset_batch(
        args.preference_model,
        completions,
        args.batch_size
    )

    # Save the dataset
    push_to_hub(preference_dataset, args.output_file)
    print(f"Dataset saved to {args.output_file}")


if __name__ == "__main__":
    main()
