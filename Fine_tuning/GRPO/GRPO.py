"""
GRPO Fine-tuning for Qwen2-VL-2B-Instruct
Finetuned for Feynman's first principles dataset
"""

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer, 
    BitsAndBytesConfig, 
    TrainingArguments,
    Qwen2VLForConditionalGeneration,
    prepare_model_for_kbit_training
)
from trl import GRPOTrainer, GRPOConfig
import wandb
from huggingface_hub import login
from peft import LoraConfig, get_peft_model
import json
import os
from rewards_func import reward_opening_hook, detailed_reward_analysis, reward_with_feedback

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "mps"
print(f"Using device: {device}")

model_name = "KhushalM/Qwen2-VL-2B-Instruct-SFT"
output_dir = "./grpo_results"
hub_model_id = "KhushalM/Qwen2-VL-2B-Instruct-GRPO-FirstPrinciples"
dataset_path = "../dataset/structured_dataset.json"

login()

wandb.login()
wandb.init(
    project="qwen2-vl-2b-instruct-grpo-first-principles",
    config={
        "model": model_name,
        "task": "First Principles Explanations",
        "reward_strategy": "Multi-component First Principles Reward",
        "lora_r": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-5,
        "batch_size": 2,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 3,
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

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

#Prepare model for training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False
print("Model prepared for kbit training")

#LoRA Setup
peft_config = LoraConfig(
    r = 32,
    lora_alpha = 64,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout = 0.05,
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
print("Model converted to LoRA")
print("Trainable Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Total Parameters: ", sum(p.numel() for p in model.parameters()))

#Load dataset
print("Loading dataset...")
with open(dataset_path, "r") as f:
    dataset = json.load(f)

dataset = Dataset.from_list(dataset)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset["train"]
val_dataset = dataset["test"]

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(val_dataset)}")

# Dataset formatting function
def format_dataset(sample):
    """Format the sample for training"""
    messages = sample["messages"]
    # Extract the user question for context
    user_message = next((msg["content"] for msg in messages if msg["role"] == "user"), "")
    
    # Format as instruction-response pair
    prompt = tokenizer.apply_chat_template(
        messages[:-1],  # All messages except the last (assistant) one
        tokenize=False,
        add_generation_prompt=True
    )
    
    return {
        "prompt": prompt,
        "response": messages[-1]["content"],  # Assistant's response
        "context": user_message
    }

# Format datasets
train_formatted = train_dataset.map(format_dataset)
val_formatted = val_dataset.map(format_dataset)
#Import reward functions
print("Loading reward functions...")
def compute_reward(responses, contexts=None):
    """
    Compute the reward for the response
    """
    if isinstance(responses, str):
        responses = [responses]
    if contexts is None:
        contexts = [None] * len(responses)
    elif isinstance(contexts, str):
        contexts = [contexts]
    
    rewards = []
    for response, context in zip(responses, contexts):
        reward = reward_with_feedback(response, context)
        rewards.append(reward)
    return rewards

# GRPO Config
GRPO_CONFIG = GRPOConfig(
    output_dir = output_dir,
    num_train_epochs = 3,
    per_device_train_batch_size = 2,
    per_device_eva_batch_size = 2,
    gradient_accumulation_steps = 8,
    learning_rate = 1e-5,
    lr_scheduler_type = "cosine",
    warmup_ratio = 0.1,
    logging_steps = 10,
    eval_steps = 50,
    save_steps = 50,
    eval_strategy ="steps",
    save_strategy="steps",
    save_total_limit = 3,
    load_best_model_at_end = True,
    metric_for_best_model = "eval_reward",
    greater_is_better = True,
    report_to = "wandb",
    push_to_hub = True,
    hub_model_id = hub_model_id,
    hub_strategy = "every_save",
    dataloader_num_workers = 2,
    remmove_unsued_columns = False,
    #GRPO specific
    max_new_tokens = 1024,
    num_generations = 4,
    temperature = 0.7,
    kl_penalty = "kl"
    kl_coef = 0.05,
    reward_model_path = None,
    bf16 = True if device == "cuda" else False,
)

print("Initializing GRPO Trainer...")
GRPO_Trainer = GRPOTrainer(
    model = model,
    tokenizer= tokenizer,
    args = GRPO_CONFIG,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    reward_function = compute_reward,
    data_collator = None,
)

print("GRPO Trainer initialized")
print(f"Number of training examples: {len(train_formatted)}")
print(f"Number of validation examples: {len(val_formatted)}")

# Custom callback for detailed logging
class RewardLoggingCallback:
    def __init__(self):
        self.step_count = 0
    
    def on_log(self, logs):
        """Log detailed reward analysis periodically"""
        if self.step_count % 50 == 0:  # Every 50 steps
            # Sample a few responses for detailed analysis
            sample_responses = logs.get('sample_responses', [])
            if sample_responses:
                for i, response in enumerate(sample_responses[:3]):  # First 3 samples
                    scores = detailed_reward_analysis(response)
                    wandb.log({
                        f"detailed_reward_sample_{i}/analogy_quality": scores['analogy_quality'],
                        f"detailed_reward_sample_{i}/step_by_step": scores['step_by_step'],
                        f"detailed_reward_sample_{i}/fundamental_concepts": scores['fundamental_concepts'],
                        f"detailed_reward_sample_{i}/engagement": scores['engagement'],
                        f"detailed_reward_sample_{i}/clarity": scores['clarity'],
                        f"detailed_reward_sample_{i}/completeness": scores['completeness'],
                        f"detailed_reward_sample_{i}/avoid_jargon": scores['avoid_jargon'],
                        f"detailed_reward_sample_{i}/total": scores['total'],
                    })
        self.step_count += 1

# Add callback
reward_callback = RewardLoggingCallback()

# Start training
print("Starting GRPO training...")
print("This will optimize for first principles explanations using our custom reward function")

try:
    trainer.train()
    print("Training completed successfully!")
    
    # Save the final model
    print("Saving final model...")
    trainer.save_model(os.path.join(output_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final_model"))


    # Test the model with a sample prompt
    print("\nTesting trained model...")
    test_prompt = "Why do objects fall to the ground when dropped?"
    
    test_messages = [
        {"role": "system", "content": "You are an expert educator who explains concepts from first principles like Richard Feynman. Start with fundamental truths, use simple analogies, and avoid jargon. Use a storytelling tone and follow a step by step explanation style:"},
        {"role": "user", "content": test_prompt}
    ]
    
    formatted_test = tokenizer.apply_chat_template(
        test_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(formatted_test, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    assistant_response = response.split("<|im_start|>assistant\n")[-1]
    
    print(f"Test Question: {test_prompt}")
    print(f"Model Response: {assistant_response}")
    
    # Evaluate the response
    reward_score, feedback = reward_with_feedback(assistant_response)
    print(f"Reward Score: {reward_score:.3f}")
    print(f"Feedback: {feedback}")
    
    # Log final test results
    wandb.log({
        "final_test/reward_score": reward_score,
        "final_test/response_length": len(assistant_response.split()),
    })

except Exception as e:
    print(f"Training failed with error: {e}")
    raise e

finally:
    wandb.finish()

print("GRPO training process completed!")