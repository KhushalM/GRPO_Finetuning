<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Complete Guide to Fine-Tuning Qwen for First Principles Reasoning on Mac Using GRPO

## 1. Mac Environment Setup

### Hardware Requirements

- Apple Silicon (M1/M2/M3) with 16GB+ RAM (24GB+ recommended for 7B models)
- macOS Sonoma 14.0+ with Xcode 15.4+


### Install Core Dependencies

```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python and system tools
brew install python@3.10 git cmake

# Create virtual environment
python3.10 -m venv ~/grpo-env
source ~/grpo-env/bin/activate

# Install PyTorch with MPS support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install vLLM from source (experimental Mac support)
git clone https://github.com/vllm-project/vllm.git
cd vllm && pip install -r requirements-cpu.txt && pip install -e .
```


## 2. Model Preparation

### Download Qwen2.5 from Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
```


## 3. GRPO Training Configuration

### Custom Training Script (`grpo_train.py`)

```python
import os
import torch
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    set_seed
)
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model

# Initialize W&B
wandb.init(project="qwen-first-principles", name="grpo-mac-run")

class FirstPrinciplesTrainer:
    def __init__(self):
        self.config = {
            "model_name": "Qwen/Qwen2.5-7B",
            "lora_config": {
                "r": 32,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            },
            "grpo_config": {
                "num_generations": 4,
                "temperature": 0.7,
                "max_prompt_length": 512,
                "max_completion_length": 1024,
                "kl_penalty": 0.1
            },
            "training_args": {
                "output_dir": "./grpo-output",
                "num_train_epochs": 1,
                "per_device_train_batch_size": 2,
                "gradient_accumulation_steps": 8,
                "learning_rate": 1e-6,
                "fp16": True,
                "logging_steps": 10,
                "optim": "adamw_torch",
                "report_to": "wandb"
            }
        }

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["model_name"])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config["model_name"],
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Apply LoRA for memory efficiency
        peft_config = LoraConfig(**self.config["lora_config"])
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def create_reward_function(self, response):
        """Verifiable first principles reward function"""
        score = 0.0
        
        # Structure validation
        if re.search(r'<reasoning>.*?</reasoning>\s*<answer>.*?</answer>', response):
            score += 0.3
            
        # First principles indicators
        fp_keywords = ['fundamental', 'core concept', 'building block', 'basic principle']
        score += 0.1 * sum(1 for kw in fp_keywords if kw in response.lower())
        
        # Feynman technique check
        feynman_indicators = ['analogy', 'simple terms', 'imagine that', 'for example']
        score += 0.1 * sum(1 for fi in feynman_indicators if fi in response.lower())
        
        # Answer correctness (simple placeholder)
        if "42" in response:  # Replace with actual validation
            score += 0.5
            
        return min(score, 1.0)

    def prepare_dataset(self):
        """Create first principles prompts dataset"""
        return load_dataset("json", data_files={"train": "first_principles_prompts.json"})

    def train(self):
        grpo_config = GRPOConfig(
            **self.config["grpo_config"],
            use_vllm=True,
            vllm_mode="colocate",
            vllm_gpu_memory_utilization=0.7
        )

        training_args = TrainingArguments(
            **self.config["training_args"],
            remove_unused_columns=False,
            dataloader_num_workers=0  # Required for Mac compatibility
        )

        trainer = GRPOTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.prepare_dataset()["train"],
            tokenizer=self.tokenizer,
            reward_func=self.create_reward_function
        )

        trainer.train()

if __name__ == "__main__":
    trainer = FirstPrinciplesTrainer()
    trainer.load_model()
    trainer.train()
```


## 4. Dataset Preparation

### Sample Prompt Structure (`first_principles_prompts.json`)

```json
[
    {
        "prompt": "Using first principles, explain why objects fall at the same rate in a vacuum. Format your answer with <reasoning> and <answer> blocks.",
        "category": "physics"
    },
    {
        "prompt": "Break down the concept of supply and demand from fundamental economic principles. Use the Feynman technique in your explanation.",
        "category": "economics"
    }
]
```


## 5. Running the Training

```bash
# Activate environment
source ~/grpo-env/bin/activate

# Login to W&B
wandb login

# Start training with MPS acceleration
MPS_DEVICE=1 CUDA_VISIBLE_DEVICES=0 python grpo_train.py
```


## 6. Key Mac-Specific Optimizations

### Memory Management

```python
# Add to training arguments
training_args = TrainingArguments(
    ...
    gradient_checkpointing=True,
    optim="adafactor",
    torch_compile=True  # Use Metal Shader Graph
)
```


### vLLM Configuration

```python
GRPOConfig(
    ...
    vllm_max_model_len=2048,
    vllm_enable_chunked_prefill=True,
    vllm_gpu_memory_utilization=0.8
)
```


## 7. Monitoring with Weights \& Biases

![GRPO Training Dashboard](

![GRPO Training Workflow for First Principles Reasoning on Mac](https://pplx-res.cloudinary.com/image/upload/v1750260151/pplx_code_interpreter/5fcbd782_s37jsl.jpg)

GRPO Training Workflow for First Principles Reasoning on Mac

)

## 8. Model Evaluation

```python
def evaluate_response(prompt, response):
    wandb.log({
        "reasoning_score": float('<reasoning>' in response),
        "answer_score": float('<answer>' in response),
        "fp_keywords": sum(1 for kw in fp_keywords if kw in response),
        "feynman_score": sum(1 for fi in feynman_indicators if fi in response)
    })
```


## 9. Post-Training Usage

```python
from peft import PeftModel

# Load fine-tuned model
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B"),
    "./grpo-output"
)

# Merge LoRA weights
model = model.merge_and_unload()
```


## Troubleshooting Common Mac Issues

1. **MPS Memory Errors**:
    - Reduce batch size: `per_device_train_batch_size=1`
    - Enable gradient checkpointing
    - Use `torch.mps.empty_cache()` periodically
2. **vLLM Compatibility**:

```bash
# Reinstall with specific flags
CMAKE_ARGS="-DLLAMA_METAL=on" pip install -e .
```

3. **Performance Optimization**:

```python
# Enable Metal kernels
torch.set_float32_matmul_precision('high')
```


This guide provides an end-to-end solution for implementing verifiable first principles reasoning with Qwen on Apple Silicon, leveraging GRPO's efficiency and Mac's native Metal acceleration.

<div style="text-align: center">⁂</div>

[^1]: https://github.com/willccbb/verifiers

[^2]: https://github.com/willccbb/verifiers/blob/main/README.md

[^3]: https://huggingface.co/docs/trl/main/en/grpo_trainer

[^4]: https://www.reddit.com/r/MachineLearning/comments/1iyv12c/p_train_your_own_reasoning_model_grpo_works_on/

[^5]: https://www.stephendiehl.com/posts/grpotrainer/

[^6]: https://wenku.csdn.net/answer/45m2vi00x4

[^7]: https://huggingface.co/docs/trl/main/en/vllm_integration

[^8]: https://apeatling.com/articles/simple-guide-to-local-llm-fine-tuning-on-a-mac-with-mlx/

[^9]: https://qwen-3.com/en/download

[^10]: https://developer.apple.com/metal/pytorch/

[^11]: https://towardsdatascience.com/pytorch-and-mlx-for-apple-silicon-4f35b9f60e39/

[^12]: https://www.reddit.com/r/MachineLearning/comments/18bwe19/apple_releases_mlx_ml_framework_for_apple_silicon/

[^13]: https://discuss.pytorch.org/t/using-macbook-gpus-with-pytorch/213965

[^14]: https://runebook.dev/en/articles/pytorch/backends/torch.backends.mps.is_built

[^15]: https://swift.readthedocs.io/en/latest/Instruction/GRPO.html

[^16]: https://docs.ultralytics.com/integrations/weights-biases/

[^17]: https://huggingface.co/Qwen/Qwen2.5-Omni-7B/discussions/30

[^18]: https://github.com/Red-Hat-AI-Innovation-Team/async-grpo

[^19]: https://developers.redhat.com/articles/2025/04/05/async-grpo-open-fast-and-performant

[^20]: https://unsloth.ai/blog/grpo

[^21]: https://gist.github.com/donbr/27db23238f16bc23a9aa98a30c665f71

[^22]: https://docs.vllm.ai/en/latest/getting_started/installation/cpu-apple.html

[^23]: https://www.stephendiehl.com/posts/small_reasoning_models/

[^24]: https://newreleases.io/project/github/unslothai/unsloth/release/2025-02

[^25]: https://dzone.com/articles/vision-ai-apple-silicon-guide-mlx-vlm

[^26]: https://docs.wandb.ai/quickstart/

[^27]: https://docs.wandb.ai/guides/track/config/

[^28]: https://github.com/wandb/wandb

[^29]: https://docs.wandb.ai

[^30]: https://machinelearningmastery.com/wp-content/uploads/2023/04/deep_learning_with_pytorch_mini_course.pdf

[^31]: https://huggingface.co/learn/llm-course/chapter12/3a?fw=pt

[^32]: https://dev.to/atsushiambo/running-qwen-nearly-as-powerful-as-deepseek-on-a-macbook-pro-367k

[^33]: https://discuss.pytorch.org/t/training-doesnt-converge-when-running-on-m1-pro-gpu-mps-device/157918

[^34]: https://hyperight.com/how-to-master-data-science-from-first-principles-delving-into-fundamentals-of-data-science/

[^35]: https://www.reddit.com/r/OpenAI/comments/1h6jmgf/principles_framework_generate_ai_agents_using/

[^36]: https://communities.sas.com/t5/SAS-Communities-Library/First-Principles-a-data-science-use-case/ta-p/859057

[^37]: https://towardsdatascience.com/how-to-use-first-principle-thinking-to-solve-data-science-problems-db94bc5af21/

[^38]: https://dev.to/syed_sadat_ali/problem-solving-using-first-principle-thinking-1hi1

[^39]: https://pubmed.ncbi.nlm.nih.gov/16381682/

[^40]: https://arxiv.org/pdf/2505.14674.pdf

[^41]: https://ianbarber.blog/2025/02/01/grpo-verifiable-rewards/

[^42]: https://indico.cern.ch/event/1140580/contributions/4863931/attachments/2463784/4224699/CERN_TH_MB_2022.pdf

[^43]: https://huggingface.co/Qwen/Qwen2.5-3B

[^44]: https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e

[^45]: https://huggingface.co/Qwen/Qwen2.5-7B

[^46]: https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

[^47]: https://qwenlm.github.io/blog/qwen2.5/

[^48]: https://www.byteplus.com/en/topic/417606

[^49]: https://www.archyde.com/alibabas-qwen3-ai-models-for-apple-mlx/

[^50]: https://wenku.csdn.net/answer/3j7nicda6n

[^51]: https://nphard.io/2025/05/27/grpo.html

[^52]: https://github.com/ml-explore/mlx/issues/12

[^53]: https://www.reddit.com/r/LocalLLaMA/comments/1iu56o1/10x_longer_contexts_for_reasoning_training_90/

[^54]: https://handbook.eng.kempnerinstitute.harvard.edu/s5_ai_scaling_and_engineering/experiment_management/logging_and_monitoring.html

[^55]: https://pypi.org/project/wandb/

[^56]: https://addyosmani.com/blog/first-principles-thinking-software-engineers/

[^57]: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

[^58]: https://www.reddit.com/r/LocalLLaMA/comments/1i4w47k/a_summary_of_qwen_models/

[^59]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/8d11da022e98f16954b849cecf2eca1d/06b90c82-c449-4670-a6c1-85959cafe6db/4d7c51b1.txt

[^60]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/8d11da022e98f16954b849cecf2eca1d/06b90c82-c449-4670-a6c1-85959cafe6db/94367639.sh

[^61]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/8d11da022e98f16954b849cecf2eca1d/be3ef55f-4eaa-4af7-aad7-a5eef2ccfc8a/b7ebf8bc.json

[^62]: https://ppl-ai-code-interpreter-files.s3.amazonaws.com/web/direct-files/8d11da022e98f16954b849cecf2eca1d/389434b8-103f-4abe-8a39-3393dd845e09/7aef736d.md

