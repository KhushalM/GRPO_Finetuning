from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import verifiers as vf
import os
import wandb
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
login()

dataset = load_dataset("json", data_files="/Users/khushal/Documents/GitHub/GRPO_Finetuning/Fine_tuning/dataset/dataset.json", split="train")
model_name = "Qwen/Qwen2-VL-2B-Instruct"

wandb.login()
wandb.init(
    project="qwen-sft-lora",
    config={
        "model": "Qwen/Qwen2-VL-2B-Instruct",
        "dataset": "your-dataset-id",
        "lora_r": 64,
        "batch_size": 2,
        "learning_rate": 2e-4
    }
)

# Skip 4-bit quantization on macOS due to bitsandbytes compatibility issues
# Load model without quantization for macOS/Apple Silicon compatibility
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    device_map="auto",
    torch_dtype=torch.bfloat16  # Use bfloat16 for memory efficiency
)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    )
tokenizer.pad_token = tokenizer.eos_token

model = prepare_model_for_kbit_training(model)
model.config.use_cache = False

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, peft_config)

#Load dataset
dataset = load_dataset("../dataset/dataset.json", split="train")

def format_dataset(sample):
    return tokenizer.apply_chat_template(
        sample["messages"],
        tokenize = False,
        add_generation_prompt=False
    )

training_args = SFTConfig(
    output_dir = "./sft_results",
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    gradient_checkpointing = True,
    learning_rate = 2e-4,
    num_train_epochs = 5,
    weight_decay = 0.01,
    report_to = "wandb",
    save_strategy = "epoch",
    save_total_limit = 1,
    logging_steps = 1,
    save_only_model = True,
    log_on_each_node = True,
    push_to_hub = True,
    hub_model_id = "KhushalM/Qwen2-VL-2B-Instruct-SFT",
    max_length = 2048,
    packing=True,
    dataset_text_field = "messages",
    bf16 = True,
    optim = "adamw_torch",  # Use regular AdamW instead of 8bit version
)

trainer = SFTTrainer(
    model = model,
    args = training_args,
    train_dataset = dataset,
    formatting_func = format_dataset,
)

trainer.train()
model.save_pretrained("./sft_results/final_model")
tokenizer.save_pretrained("./sft_results/final_model")
wandb.finish()