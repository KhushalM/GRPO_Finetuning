"""
GRPO Fine-tuning for Qwen2-VL-2B-Instruct
Finetuned for Feynman's first principles dataset
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, prepare_model_for_kbit_training
from trl import GRPOTrainer, GRPOConfig, Qwen2VLForConditionalGeneration
import wandb
from huggingface_hub import login
import vf
from reward_funcs import *
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import json

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "mps"
print(f"Using device: {device}")

model_name = "KhushalM/Qwen2-VL-2B-Instruct-SFT"
output_dir = "./grpo_results"
hub_model_id = "KhushalM/Qwen2-VL-2B-Instruct-GRPO-RLVR"
dataset_path = "/Users/khushal/Documents/GitHub/GRPO_Finetuning/Fine_tuning/dataset/structured_dataset.json"

login()

wandb.login()
wandb.init(
    project="qwen2-vl-2b-instruct-grpo",
    config={
        "model": model_name,
        "task": "Feynman's first principles",
        "reward_strategy": "First Principles + Analogies",
    }
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model = model.to(device)

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
    bias = "none",
    task_type = "CAUSAL_LM"
)

model = get_peft_model(model, peft_config)
print("Model converted to LoRA")
print("Trainable Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
print("Total Parameters: ", sum(p.numel() for p in model.parameters()))

#Load dataset

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
    """Format the dataset for chat template"""
    return tokenizer.apply_chat_template(
        sample["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
#Import reward functions

GRPO_CONFIG = GRPOConfig(

)

GRPO_Trainer = GRPOTrainer(
)

print("GRPO Trainer initialized")
print("Number of training examples: ", len(train_dataset))
print("Number of validation examples: ", len(val_dataset))

print("Training...")

GRPO_Trainer.train()

print("Training complete")

print("Saving model...")

GRPO_Trainer.save_pretrained(output_dir)