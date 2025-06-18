#!/usr/bin/env python3
"""
Verification script for GRPO fine-tuning environment setup
"""

import sys
import torch
import transformers
import trl
import peft
import datasets
import wandb
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def verify_environment():
    """Verify that all required packages are installed and working"""
    
    print("🔍 Verifying GRPO Fine-tuning Environment")
    print("=" * 50)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Check key packages
    packages = {
        "PyTorch": torch.__version__,
        "Transformers": transformers.__version__,
        "TRL": trl.__version__,
        "PEFT": peft.__version__,
        "Datasets": datasets.__version__,
        "WandB": wandb.__version__,
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
    }
    
    # Add Verifiers if available
    try:
        import verifiers as vf
        packages["Verifiers"] = vf.__version__ if hasattr(vf, '__version__') else 'installed'
    except:
        pass
    
    print("\n📦 Package Versions:")
    for package, version in packages.items():
        print(f"  ✅ {package}: {version}")
    
    # Check Mac GPU support
    print(f"\n🖥️  Mac GPU Support:")
    print(f"  MPS Available: {torch.backends.mps.is_available()}")
    print(f"  MPS Built: {torch.backends.mps.is_built()}")
    
    # Test basic tensor operations on MPS
    if torch.backends.mps.is_available():
        try:
            device = torch.device("mps")
            x = torch.randn(3, 3, device=device)
            y = torch.matmul(x, x.t())
            print(f"  ✅ MPS tensor operations working")
        except Exception as e:
            print(f"  ❌ MPS tensor operations failed: {e}")
    
    # Test model loading (small model for testing)
    print(f"\n🤖 Testing Model Loading:")
    try:
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        print("  ✅ Tokenizer loading successful")
        
        # Note: Not loading full model to save time/memory in verification
        print("  ✅ Model imports successful")
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
    
    # Check GRPO components
    print(f"\n🚀 GRPO Components:")
    try:
        from trl import GRPOConfig, GRPOTrainer
        print("  ✅ TRL GRPO imports successful")
    except Exception as e:
        print(f"  ❌ TRL GRPO imports failed: {e}")
    
    try:
        from peft import LoraConfig, get_peft_model
        print("  ✅ PEFT/LoRA imports successful")
    except Exception as e:
        print(f"  ❌ PEFT/LoRA imports failed: {e}")
    
    # Check Verifiers framework
    print(f"\n🔬 Verifiers Framework:")
    try:
        import verifiers as vf
        print(f"  ✅ Verifiers v{vf.__version__ if hasattr(vf, '__version__') else 'installed'} imported successfully")
        
        # Test key Verifiers components
        vf.GRPOTrainer
        print("  ✅ Verifiers GRPOTrainer available")
        
        vf.XMLParser
        print("  ✅ XML Parser available")
        
        vf.Rubric
        print("  ✅ Rubric system available")
        
        vf.SingleTurnEnv
        print("  ✅ Single-turn environment available")
        
    except Exception as e:
        print(f"  ❌ Verifiers framework failed: {e}")
    
    # Check vLLM (optional)
    print(f"\n🚂 vLLM (Optional):")
    try:
        import vllm
        print(f"  ✅ vLLM v{vllm.__version__ if hasattr(vllm, '__version__') else 'installed'} imported successfully")
        print("  ⚠️  Note: vLLM has experimental Mac support, may fall back to CPU")
    except Exception as e:
        print(f"  ❌ vLLM import failed: {e}")
    
    # Check reasoning-gym (optional)
    print(f"\n🏃 Reasoning Gym (Optional):")
    try:
        import reasoning_gym
        print("  ✅ Reasoning Gym imported successfully")
    except Exception as e:
        print(f"  ❌ Reasoning Gym import failed: {e}")
    
    print(f"\n🎉 Environment verification complete!")
    print(f"\nTo activate this environment in the future, run:")
    print(f"  source grpo-env/bin/activate")
    print(f"  # or use the convenience script:")
    print(f"  ./activate_env.sh")

if __name__ == "__main__":
    verify_environment() 