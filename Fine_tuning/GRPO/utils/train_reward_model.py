import json 
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from trl import RewardTrainer, RewardConfig
import torch
from huggingface_hub import login, HfFolder
from datasets import load_dataset
import wandb
import os
import dotenv
import logging
logger = logging.getLogger(__name__)

def flatten_messages(messages):
    logger.info(f"Flattening messages: {messages}")
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)

def convert_to_reward_format(preference_item):
    prompt = flatten_messages(preference_item["messages"])
    chosen = preference_item["Response A"] if preference_item["Preference"] == "A" else preference_item["Response B"]
    rejected = preference_item["Response A"] if preference_item["Preference"] == "B" else preference_item["Response B"]
    logger.info(f"Converted to reward format: {prompt}, {chosen}, {rejected}")
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

def load_and_convert_dataset(json_path):
    logger.info(f"Loading dataset from {json_path}")
    with open(json_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list([convert_to_reward_format(item) for item in data])

def load_dataset_from_hub(dataset_name):
    dataset = load_dataset(dataset_name)
    return dataset["train"]

def load_reward_model(base_model="KhushalM/Qwen2.5-1.5-SFT-Merged"):
    tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        base_model,
        num_labels=1,
        problem_type="regression",
        pad_token_id=tokenizer.pad_token_id,
    )
    model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config)
    return model, tokenizer


def train_reward_model(dataset_name, output_dir = "Qwen2.5-1.5-Feynman-Reward-Model", base_model = "KhushalM/Qwen2.5-1.5-SFT-Merged"):
    #dataset = load_and_convert_dataset(dataset_path)
    dataset = load_dataset_from_hub(dataset_name)
    model, tokenizer = load_reward_model(base_model)
    model.gradient_checkpointing_enable()
    model = torch.compile(model)

    config = RewardConfig(
        output_dir = output_dir,
        per_device_train_batch_size = 8,
        gradient_accumulation_steps = 2,
        learning_rate = 5e-6,
        max_length = 1024,
        num_train_epochs = 3,
        logging_steps = 10,
        save_strategy = "epoch",
        report_to = ["wandb"],
        remove_unused_columns = False,
        fp16 = True,
    )

    trainer = RewardTrainer(
        model = model,
        processing_class = tokenizer,
        args = config,
        train_dataset = dataset
    )
    logger.info(f"Training reward model")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Pushing reward model to Hugging Face")
    login()
    logger.info(f"Pushing reward model to Hugging Face")
    trainer.push_to_hub("KhushalM/Qwen2.5-1.5-Feynman-Reward-Model")

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(project="Qwen2.5-1.5-Feynman-Reward-Model")
    train_reward_model(
        dataset_name = "KhushalM/Qwen2.5-rlaif-Feynman-Dataset",
        output_dir = "reward_model",
        base_model = "KhushalM/Qwen2.5-1.5-SFT-Merged"
    )