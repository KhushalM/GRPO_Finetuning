from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from huggingface_hub import login
import torch
import argparse
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

repo_id   = "KhushalM/Qwen2.5-1.5BSFT"
subfolder = "final_model"
out_dir   = "Qwen2.5-1.5-SFT-Merged"

def load_sft_adapters(adapter_name):
    peft_config = PeftConfig.from_pretrained(adapter_name, subfolder=subfolder)
    logger.info(f"Loaded SFT adapter from {adapter_name}")
    return peft_config

def load_base_model(peft_config):
    base_name = peft_config.base_model_name_or_path
    base_model = AutoModelForCausalLM.from_pretrained(
        base_name,
        trust_remote_code=True,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
    )
    logger.info(f"Loaded base model from {base_name}")
    return base_model

def merge_models(adapter_name):
    base_model = load_base_model(load_sft_adapters(adapter_name))
    base_name = load_sft_adapters(adapter_name).base_model_name_or_path
    lora_model = PeftModel.from_pretrained(
        base_model,
        repo_id,
        subfolder=subfolder,
        device_map={"": "cpu"},
        torch_dtype=torch.float32,
    )
    logger.info(f"Loaded LoRA model from {adapter_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    output_dir = "Qwen2.5-1.5-SFT-Merged"
    start_time = time.time()
    merged_model = lora_model.merge_and_unload()
    end_time = time.time()
    logger.info(f"Time taken to merge models: {end_time - start_time} seconds")
    logger.info(f"Merged model from {adapter_name} and {base_name}")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info(f"Saved merged model to {output_dir}")
    # merged_model.push_to_hub(
    #     repo_id="KhushalM/Qwen2.5-1.5-SFT-Merged",
    #     use_auth_token=True,
    # )
    tokenizer.push_to_hub(
        repo_id="KhushalM/Qwen2.5-1.5-SFT-Merged",
        use_auth_token=True,
    )

def main():
    login(token=os.getenv("HF_TOKEN"))
    merge_models("KhushalM/Qwen2.5-1.5BSFT")

if __name__ == "__main__":
    main()