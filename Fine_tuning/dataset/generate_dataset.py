"""
Fixed dataset generation script that creates proper JSON arrays and handles modern OpenAI API
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

# Load environment variables from .env file
load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset for first principles")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use for generation")
    parser.add_argument("--num_prompts", type=int, default=20, help="Number of prompts to generate")
    parser.add_argument("--batch_size", type=int, default=5, help="Batch size for generation")
    parser.add_argument("--output_file", type=str, default="first_principles_dataset_fixed.json", help="File to save the dataset")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    return parser.parse_args()

SYSTEM_INSTRUCTION = (
     "You are an expert educator who excels at breaking down complex ideas from first principles. "
    "When asked to generate dataset entries, you reply **only** with valid JSON. Do not include tokens like ```json or ```"
    "Each object must have two keys: 'prompt' and 'answer'. "
    "- 'prompt' should be a short, clear instruction while some can be more detailed and complex for an LLM to *explain* a concept from first principles. Prompt should not necessarily be a direct question but also something that needs explanation. It should also not necessarily have the word 'explain' or 'explain from first principles' in it. Question could also be from an observation or context from a screenshot.\n"
    "- 'answer' must provide a step-by-step break down from first-principles explanation with examples both real and metaphorical. The answer should  clarify the fundamental concepts and simple to understand. It should be like someone explaining in Richard Feynman's style. Keep answers within 500/600 words."
)

USER_TEMPLATE = (
    "Generate {n} **unique** prompt/answer pairs that meet the above format. "
    "Cover a broad range of disciplines (STEM, microeconomics, macroeconomics, philosophy, computer science, AI, Sports, Geo-Politics, etc.) "
    "and vary difficulty from beginner to advanced. Return *only* a JSON array."
)

def extract_json_from_response(content: str) -> str:
    """Extract JSON from markdown code blocks if present"""
    content = content.strip()
    
    # Handle markdown code blocks that may wrap the JSON
    if content.startswith("```"):
        lines = content.split('\n')
        json_lines = []
        in_json_block = False
        
        for line in lines:
            if line.strip().startswith("```json") or (line.strip() == "```" and not in_json_block):
                in_json_block = True
                continue
            elif line.strip() == "```" and in_json_block:
                break
            elif in_json_block:
                json_lines.append(line)
        
        if json_lines:
            return '\n'.join(json_lines)
    
    return content

def request_pairs(model: str, n: int, temperature: float) -> list[dict]:
    """Call teacher LLM to generate a list of prompt/answer pairs"""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTION},
            {"role": "user", "content": USER_TEMPLATE.format(n=n)}
        ],
        temperature=temperature
    )

    content = response.choices[0].message.content
    
    # Extract JSON from potential markdown wrapper
    json_content = extract_json_from_response(content)

    try:
        dataset = json.loads(json_content)
        assert isinstance(dataset, list), f"Response must be a JSON array, got {type(dataset)}"
        
        for i, obj in enumerate(dataset):
            assert isinstance(obj, dict), f"Item {i} must be a JSON object, got {type(obj)}"
            assert set(obj.keys()) == {"prompt", "answer"}, f"Item {i} must have 'prompt' and 'answer' keys, got: {list(obj.keys())}"
            assert isinstance(obj["prompt"], str) and obj["prompt"].strip(), f"Item {i} prompt must be a non-empty string"
            assert isinstance(obj["answer"], str) and obj["answer"].strip(), f"Item {i} answer must be a non-empty string"
            
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed. Raw response:\n{content}\n")
        print(f"[ERROR] Extracted content:\n{json_content}\n")
        raise RuntimeError(f"Invalid JSON in teacher output: {e}") from e
    except Exception as e:
        print(f"[ERROR] Validation failed. Raw response:\n{content}\n")
        raise RuntimeError(f"Teacher output validation failed: {e}") from e
    
    return dataset

def main() -> None:
    args = parse_args()
    
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("Error: OPENAI_API_KEY is not set. Please set it in your .env file")

    print(f"🚀 Starting dataset generation")
    print(f"   Model: {args.model}")
    print(f"   Target: {args.num_prompts} prompts")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Output: {args.output_file}")
    print("-" * 50)

    generated = 0
    retries = 0
    max_retries = 3
    all_pairs = []  # Collect all pairs first, then write as JSON array

    while generated < args.num_prompts:
        remaining = args.num_prompts - generated
        batch_size = min(args.batch_size, remaining)
        
        try:
            print(f"[INFO] Requesting {batch_size} pairs...")
            pairs = request_pairs(args.model, batch_size, args.temperature)
            
            all_pairs.extend(pairs)
            generated += len(pairs)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[INFO] ✅ Generated {len(pairs)} pairs. Total: {generated}/{args.num_prompts} at {timestamp}")
            retries = 0
            
            # Rate limiting to avoid API issues
            if generated < args.num_prompts:
                time.sleep(2)
                
        except Exception as e:
            retries += 1
            if retries <= max_retries:
                sleep_time = 2 ** retries + random.uniform(0, 1)
                print(f"[WARN] ⚠️  Failed to generate pairs (attempt {retries}/{max_retries})")
                print(f"[WARN] Error: {str(e)}")
                print(f"[WARN] Retrying in {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
            else:
                print(f"\n[ERROR] ❌ Failed after {max_retries} retries")
                print(f"[ERROR] Generated {generated} pairs before failure")
                if all_pairs:
                    # Save partial results
                    partial_file = args.output_file.replace('.json', '_partial.json')
                    with open(partial_file, "w", encoding="utf-8") as f:
                        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] Saved {len(all_pairs)} pairs to {partial_file}")
                break

    # Write all pairs as a proper JSON array
    if all_pairs:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(all_pairs, f, indent=2, ensure_ascii=False)

        print(f"\n🎉 Dataset generation complete!")
        print(f"✅ Generated {len(all_pairs)} pairs")
        print(f"💾 Saved to: {args.output_file}")
        print(f"📊 Format: Proper JSON array")
        
        # Validate the output
        with open(args.output_file, 'r') as f:
            test_load = json.load(f)
        print(f"✅ Output validation passed - {len(test_load)} valid entries")
    else:
        print("❌ No data generated")

if __name__ == "__main__":
    main()
