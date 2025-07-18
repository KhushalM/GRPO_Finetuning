import json 
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
import torch

def flatten_messages(messages):
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)

def convert_to_reward_format(preference_item):
    prompt = flatten_messages(preference_item["messages"])
    chosen = preference_item["Response A"] if preference_item["Preference"] == "A" else preference_item["Response B"]
    rejected = preference_item["Response A"] if preference_item["Preference"] == "B" else preference_item["Response B"]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

def load_and_convert_dataset(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return Dataset.from_list([convert_to_reward_format(item) for item in data])

def load_reward_model(model_name = "Qwen/Qwen2.5-1.5-SFT-Merged"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

def train_reward_model(dataset, output_dir = "reward_model", base_model = "Qwen/Qwen2.5-1.5-SFT-Merged"):
    dataset = load_and_convert_dataset(dataset_path)
    model, tokenizer = load_reward_model(base_model)

    config = RewardConfig(
        output_dir = output_dir,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        learning_rate = 5e-6,
        max_length = 1024,
        num_train_epochs = 3,
        logging_steps = 10,
        evaluation_strategy = "no",
        save_strategy = "epoch",
        report_to = ["wandb"],
        remove_unused_columns = False,
    )

    trainer = RewardTrainer(
        model = model,
        tokenizer = tokenizer,
        args = config,
        train_dataset = dataset
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    train_reward_model(
        dataset_path = "data/preference_data.json",
        output_dir = "reward_model",
        base_model = "Qwen/Qwen2.5-1.5-SFT-Merged"
    )