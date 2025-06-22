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

# Load environment variables from .env file
load_dotenv()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate dataset for first principles")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini", help="Model to use for generation")
    parser.add_argument("--num_prompts", type=int, default=600, help="Number of conversation objects to generate")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch size for generation")
    parser.add_argument("--output_file", type=str, default="structured_dataset.json", help="File to save the dataset")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    return parser.parse_args()

SYSTEM_INSTRUCTION = (
    "You are an expert educator who breaks down complex ideas from first principles. \n"
    "**Reply ONLY with a valid JSON object** using **this exact structure**: \n"
    "{\n"
    "  \"messages\": [\n"
    "    {\"role\": \"system\", \"content\": \"You are an expert educator who explains concepts from first principles like Richard Feynman. "
    "Start with fundamental truths, use simple analogies, and avoid jargon. Use a storytelling tone and follow a step by step explanation style: "
    "    {\"role\": \"user\", \"content\": \"<concept observation/question>\"},\n"
    "    {\"role\": \"assistant\", \"content\": \"<Feynman-style explanation using the 3-part structure>\"}\n"
    "  ]\n"
    "}\n\n"
    "**Critical Rules**: \n"
    "1. **Never** use `prompt`/`answer` keys. \n"
    "2. **Never** include non-JSON text (e.g., ``````). \n"
    "3. **Each object must contain exactly one user-assistant exchange.** \n"
    "   - **Do not include multiple user-assistant pairs in a single object.** \n"
    "4. **Assistant content must follow this explanation structure (Do not include Step 1, Step 2, Step 3, Step 4 in the assistant message)**:\n"
    "   Step 1: Begin with an easy metaphor (e.g., 'Okay, let’s imagine...') something that draws a visual image in the user's mind, "
    "   Step 2: Explain with a methaphorical example easily understood even by a middle school student explaining from fundamentals and first principles"
    "   Step 3: Move to a real-world example grounded in daily life or observable phenomena, and build it from grounds up fundamentals fully from first principles"
    "   Step 4: End by asking the user if they now feel they’ve understood or connected the dots.\"},\n"
    "5. **Prohibited**: \n"
    "   - No phrases like 'explain from first principles' in user messages. \n"
    "   - No standalone keys like 'prompt' or 'answer'. \n"
    "   - No multi-turn conversations in a single object. \n"
    "6. **Each sample must be a standalone conversation.** \n"
    "7. **The user message should be a question or observation about the concept.** \n"
    "8. **The user message question complexity range should vary from basic concepts to advanced concepts.** \n"
    "9. **The assistant message should be limited to 600 words. Easier topics could be shorter, harder topics could be longer.** \n"
    "10. **Do not forget the system message.** \n\n"
    "11. **Item must have exactly 3 messages (system, user, assistant)** \n"
    "**JSON Formatting Rules:** \n"
    "- All content must be on single lines (no actual line breaks) \n"
    "- Use \\n for line breaks, \\t for tabs \n"
    "- Escape all quotes with backslash-quote \n"
    "- No control characters (newlines, tabs, etc.) in strings"
)



USER_TEMPLATE = (
    "Generate {n} **unique** conversation objects that meet the above format."
    "Cover a broad range of disciplines"
    "**Diversity Requirements**: "
    "1. **Discipline Coverage**: "
    "   - 15% STEM (Physics/Chemistry/Biology) "
    "   - 25% Economics (Micro/Macro/Behavioral) "
    "   - 10% Philosophy (Ethics/Epistemology) "
    "   - 25% Tech (AI/CS/Cybersecurity) "
    "   - 10% Social Sciences (Psychology/Sociology) "
    "   - 10% Geo-Politics "
    "   - 5% Sports/Arts "
    "2. **Difficulty Distribution**: "
    "   - 30% Beginner"
    "   - 50% Intermediate"
    "   - 20% Advanced"
    "3. **Question Types**: "
    "   - 40% Direct Conceptual questions"
    "   - 10% Conceptual Observations (About observations from the user's screen)"
    "   - 20% Indirect Observations (About indirect observations from the user's screen)"
    "   - 20% Scenario-based"
    "   - 10% Comparative"
    "4. **Uniqueness Safeguards**: "
    "   - No duplicate concepts"
    "   - Vary explanation angles"
    "5. **Output Format**: "
    "   - Return *only* a JSON array of objects with the 'messages' structure."
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

def request_conversations(model: str, n: int, temperature: float) -> list[dict]:
    """Call teacher LLM to generate a list of conversation objects with messages format"""
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
            assert set(obj.keys()) == {"messages"}, f"Item {i} must have 'messages' key, got: {list(obj.keys())}"
            assert isinstance(obj["messages"], list), f"Item {i} messages must be a list, got {type(obj['messages'])}"
            assert len(obj["messages"]) == 3, f"Item {i} must have exactly 3 messages (system, user, assistant), got {len(obj['messages'])}"
            
            # Validate each message
            messages = obj["messages"]
            expected_roles = ["system", "user", "assistant"]
            for j, msg in enumerate(messages):
                assert isinstance(msg, dict), f"Item {i}, message {j} must be a dict, got {type(msg)}"
                assert set(msg.keys()) == {"role", "content"}, f"Item {i}, message {j} must have 'role' and 'content' keys, got: {list(msg.keys())}"
                assert msg["role"] == expected_roles[j], f"Item {i}, message {j} must have role '{expected_roles[j]}', got '{msg['role']}'"
                assert isinstance(msg["content"], str) and msg["content"].strip(), f"Item {i}, message {j} content must be a non-empty string"
            
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
    all_conversations = []  # Collect all conversations first, then write as JSON array

    while generated < args.num_prompts:
        remaining = args.num_prompts - generated
        batch_size = min(args.batch_size, remaining)
        
        try:
            print(f"[INFO] Requesting {batch_size} conversations...")
            conversations = request_conversations(args.model, batch_size, args.temperature)
            
            all_conversations.extend(conversations)
            generated += len(conversations)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[INFO] ✅ Generated {len(conversations)} conversations. Total: {generated}/{args.num_prompts} at {timestamp}")
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
                print(f"[ERROR] Generated {generated} conversations before failure")
                if all_conversations:
                    # Save partial results
                    partial_file = args.output_file.replace('.json', '_partial.json')
                    with open(partial_file, "w", encoding="utf-8") as f:
                        json.dump(all_conversations, f, indent=2, ensure_ascii=False)
                    print(f"[INFO] Saved {len(all_conversations)} conversations to {partial_file}")
                break

    # Write all conversations as a proper JSON array
    if all_conversations:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(all_conversations, f, indent=2, ensure_ascii=False)

        print(f"\n🎉 Dataset generation complete!")
        print(f"✅ Generated {len(all_conversations)} conversations")
        print(f"💾 Saved to: {args.output_file}")
        print(f"📊 Format: Proper JSON array with messages structure")
        
        # Validate the output
        with open(args.output_file, 'r') as f:
            test_load = json.load(f)
        print(f"✅ Output validation passed - {len(test_load)} valid entries")
    else:
        print("❌ No data generated")

if __name__ == "__main__":
    main()
