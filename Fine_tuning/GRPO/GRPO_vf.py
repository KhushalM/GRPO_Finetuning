"""
GRPO Fine-tuning using Verifiers Framework for Qwen2-VL-2B-Instruct
Finetuned for Feynman's first principles dataset using verifiers library
"""

import torch
import json
import os
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Import verifiers framework
import verifiers as vf
from rewards_func import reward_opening_hook, detailed_reward_analysis, reward_with_feedback

# Device configuration
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "mps"
print(f"Using device: {device}")

# Model and dataset configuration
model_name = "KhushalM/Qwen2-VL-2B-Instruct-SFT"
output_dir = "./grpo_vf_results"
hub_model_id = "KhushalM/Qwen2-VL-2B-Instruct-GRPO-VF-FirstPrinciples"
dataset_path = "../dataset/structured_dataset.json"

# Login to services
login()
wandb.login()

# Initialize wandb with verifiers-specific config
wandb.init(
    project="qwen2-vl-2b-instruct-grpo-vf-first-principles",
    config={
        "model": model_name,
        "task": "First Principles Explanations",
        "framework": "verifiers",
        "reward_strategy": "Multi-component First Principles Reward",
        "lora_r": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
        "num_generations": 4,
        "reward_components": {
            "analogy_quality": 0.20,
            "step_by_step": 0.15,
            "fundamental_concepts": 0.20,
            "engagement": 0.15,
            "clarity": 0.15,
            "completeness": 0.10,
            "avoid_jargon": 0.05
        }
    }
)

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading model and tokenizer...")
model, tokenizer = vf.get_model_and_tokenizer(
    model_name,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map=device
)

# Prepare model for training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
print("Model prepared for kbit training")

# LoRA Setup
peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
print("Model converted to LoRA")
print("Trainable Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Total Parameters: ", sum(p.numel() for p in model.parameters()))

# Load and prepare dataset
print("Loading dataset...")
with open(dataset_path, "r") as f:
    dataset_raw = json.load(f)

# Transform dataset to verifiers format
def transform_to_verifiers_format(data):
    """Transform our dataset to verifiers-compatible format"""
    transformed = []
    for item in data:
        messages = item["messages"]
        user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
        assistant_message = next((msg["content"] for msg in messages if msg["role"] == "assistant"), "")
        
        transformed.append({
            "question": user_message,
            "answer": assistant_message,  # This will be our target for reward evaluation
            "messages": messages  # Keep original format for reference
        })
    return transformed

transformed_data = transform_to_verifiers_format(dataset_raw)
dataset = Dataset.from_list(transformed_data)
dataset = dataset.train_test_split(test_size=0.1)

print(f"Train dataset size: {len(dataset['train'])}")
print(f"Test dataset size: {len(dataset['test'])}")

# Define custom reward function for verifiers
def first_principles_reward_func(prompt, completion, answer, **kwargs):
    """
    Custom reward function that uses our existing reward logic
    
    Args:
        prompt: The input prompt/question
        completion: The model's completion/response
        answer: The target answer (from dataset)
        **kwargs: Additional arguments
    
    Returns:
        float: Reward score between 0 and 1
    """
    try:
        # Use our existing reward function
        reward_score = reward_with_feedback(completion, prompt)[0]  # Get just the score
        return reward_score
    except Exception as e:
        print(f"Error in reward function: {e}")
        return 0.0

# Create XMLParser for consistent formatting
parser = vf.XMLParser(['reasoning', 'answer'])

# Create Rubric with our custom reward function and format reward
rubric = vf.Rubric(
    first_principles_reward_func,  # Our custom reward function
    parser.get_format_reward_func(),  # Built-in format reward
    weights=[0.8, 0.2]  # 80% content quality, 20% format
)

# System prompt for first principles explanations
system_prompt = """You are an expert educator who explains concepts from first principles like Richard Feynman. 
Your goal is to make complex ideas accessible through:
1. Starting with fundamental truths and building up
2. Using simple analogies and relatable examples
3. Following a clear step-by-step reasoning process
4. Avoiding jargon and technical language
5. Engaging the reader with questions and storytelling

Respond in the following format:
<reasoning>
[Your step-by-step reasoning process, using analogies and building from fundamentals]
</reasoning>
<answer>
[Your clear, engaging explanation of the concept]
</answer>"""

# Create SingleTurnEnv for first principles explanations
print("Creating verifiers environment...")
vf_env = vf.SingleTurnEnv(
    dataset=dataset['train'],
    system_prompt=system_prompt,
    parser=parser,
    rubric=rubric
)

# GRPO training arguments using verifiers defaults
training_args = vf.grpo_defaults(
    run_name="qwen2-vl-2b-grpo-vf-first-principles",
    output_dir=output_dir,
    learning_rate=1e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    save_steps=50,
    eval_steps=50,
    logging_steps=10,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="wandb",
    push_to_hub=True,
    hub_model_id=hub_model_id,
    bf16=True if device == "cuda" else False,
    # GRPO specific parameters
    num_generations=4,
    max_completion_length=1024,
    temperature=0.7,
    kl_coef=0.05,
)

# Initialize GRPO Trainer using verifiers
print("Initializing Verifiers GRPO Trainer...")
trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args
)

print("Verifiers GRPO Trainer initialized")
print("This trainer uses the verifiers framework for more flexible and efficient GRPO training")

# Custom callback for detailed logging
class VerifiersRewardLoggingCallback:
    def __init__(self):
        self.step_count = 0
    
    def on_log(self, logs):
        """Log detailed reward analysis periodically"""
        if self.step_count % 50 == 0:  # Every 50 steps
            print(f"Step {self.step_count}: Logging detailed reward metrics")
            # Add any custom logging here
        self.step_count += 1

# Add callback
reward_callback = VerifiersRewardLoggingCallback()

# Test the environment before training
print("\nTesting environment with sample data...")
try:
    # Get a sample from the dataset
    sample_data = dataset['train'][0]
    print(f"Sample question: {sample_data['question'][:100]}...")
    
    # Test the reward function
    test_completion = "This is a test completion for first principles explanation."
    test_reward = first_principles_reward_func(
        sample_data['question'], 
        test_completion, 
        sample_data['answer']
    )
    print(f"Test reward score: {test_reward:.3f}")
    print("Environment testing completed successfully!")
    
except Exception as e:
    print(f"Environment test failed: {e}")
    print("Continuing with training anyway...")

# Start training
print("\nStarting Verifiers GRPO training...")
print("This will optimize for first principles explanations using verifiers framework")

try:
    trainer.train()
    print("Training completed successfully!")
    
    # Save the final model
    print("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))

    # Test the trained model
    print("\nTesting trained model...")
    test_prompt = "Why do objects fall to the ground when dropped?"
    
    # Use verifiers for evaluation
    test_results = vf_env.evaluate(
        client=None,  # Using local model
        model=model,
        num_samples=1,
        test_prompts=[test_prompt]
    )
    
    print(f"Test Question: {test_prompt}")
    print(f"Model Response: {test_results['completions'][0] if test_results['completions'] else 'No response'}")
    print(f"Reward Score: {test_results['rewards_avg']:.3f}")
    
    # Log final test results
    wandb.log({
        "final_test/reward_score": test_results['rewards_avg'],
        "final_test/num_samples": len(test_results['completions']),
    })

except Exception as e:
    print(f"Training failed with error: {e}")
    import traceback
    traceback.print_exc()
    raise e

finally:
    wandb.finish()

print("Verifiers GRPO training process completed!")
print(f"Model saved to: {output_dir}")
print("Key advantages of using verifiers framework:")
print("- More efficient GRPO implementation")
print("- Better support for custom environments")
print("- Cleaner abstraction for reward functions")
print("- Built-in support for format checking")
print("- Easier integration with evaluation pipelines") 