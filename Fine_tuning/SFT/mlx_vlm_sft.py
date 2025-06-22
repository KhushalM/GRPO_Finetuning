#!/usr/bin/env python3
"""
MLX-VLM Fine-tuning Script for Qwen2-VL
Optimized for Apple Silicon using MLX framework
"""

import json
import os
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from datasets import load_dataset
import wandb
from huggingface_hub import login

# Login to HuggingFace
login()

# Initialize Weights & Biases
wandb.login()
wandb.init(
    project="qwen2-vl-mlx-sft",
    config={
        "model": "mlx-community/Qwen2-VL-2B-Instruct-4bit",
        "framework": "MLX-VLM",
        "batch_size": 1,
        "learning_rate": 2e-4,
        "lora_rank": 32,
        "lora_alpha": 64,
        "platform": "Apple Silicon"
    }
)

def convert_dataset_to_jsonl(dataset_path):
    """Convert JSON dataset to JSONL format suitable for MLX-VLM"""
    
    # Load your dataset (already in chat format)
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    # Save as JSONL format
    output_path = "first_principles_dataset.jsonl"
    with open(output_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')
    
    print(f"Dataset converted to JSONL and saved to {output_path}")
    print(f"Dataset contains {len(data)} training examples")
    return output_path

def main():
    # Model configuration
    model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
    dataset_path = "../dataset/first_principles_dataset.json"
    
    print("Dataset is already in chat format, converting to JSONL...")
    chat_dataset_path = convert_dataset_to_jsonl(dataset_path)
    
    print("Loading model and processor...")
    model, processor = load(model_path)
    config = load_config(model_path)
    
    print("Model loaded successfully with MLX!")
    print(f"Model type: {type(model)}")
    
    # Test the model with a simple example
    print("\nTesting model with a sample prompt...")
    test_prompt = "Explain the concept of machine learning in simple terms."
    
    # Apply chat template
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": test_prompt}
    ]
    
    formatted_prompt = apply_chat_template(
        processor, config, test_prompt, num_images=0
    )
    
    print("Formatted prompt:", formatted_prompt[:200] + "...")
    
    # Generate a test response
    print("\nGenerating test response...")
    try:
        response = generate(
            model, 
            processor, 
            prompt=formatted_prompt, 
            image=None,
            verbose=True,
            max_tokens=100,
            temp=0.7
        )
        print("Test generation successful!")
        print("Response:", response)
    except Exception as e:
        print(f"Error during generation: {e}")
    
    # MLX-VLM LoRA training command
    print("\n" + "="*50)
    print("MLX-VLM Training Configuration")
    print("="*50)
    
    training_command = f"""
    To start fine-tuning with MLX-VLM, run:
    
    python -m mlx_vlm.lora \\
        --model-path {model_path} \\
        --dataset {chat_dataset_path} \\
        --batch-size 1 \\
        --learning-rate 2e-4 \\
        --epochs 3 \\
        --lora-rank 32 \\
        --lora-alpha 64 \\
        --lora-dropout 0.05 \\
        --output-path ./mlx_sft_results \\
        --apply-chat-template \\
        --print-every 10
    
    This will:
    - Use the 4-bit quantized Qwen2-VL model
    - Train with LoRA for efficient fine-tuning
    - Save adapters to ./mlx_sft_results
    - Apply chat template formatting
    - Work efficiently on Apple Silicon
    """
    
    print(training_command)
    
    # Log configuration to wandb
    wandb.log({
        "model_loaded": True,
        "dataset_size": len(json.load(open(dataset_path))),
        "converted_dataset": chat_dataset_path
    })
    
    wandb.finish()

if __name__ == "__main__":
    main() 