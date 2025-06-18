#!/bin/bash
# Activation script for GRPO fine-tuning environment

echo "Activating GRPO fine-tuning environment..."
source grpo-env/bin/activate

# Set environment variables for Mac optimization
export MPS_DEVICE=1
export CUDA_VISIBLE_DEVICES=0
export OPENAI_API_KEY=sk-proj-1234567890
export WANDB_API_KEY=1234567890
export PTORCH_ENABLE_MPS_FALLBACK=1

echo "Environment activated!"
echo "Python version: $(python -c 'import sys; print(sys.version.split()[0])')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "MPS available: $(python -c 'import torch; print(torch.backends.mps.is_available())')"
echo "Verifiers version: $(python -c 'import verifiers as vf; print(vf.__version__ if hasattr(vf, "__version__") else "installed")')"
echo ""
echo "Ready for GRPO fine-tuning with Verifiers!"

# Check if .env file exists and has API key
if [ -f ".env" ]; then
    if grep -q "OPENAI_API_KEY=your-actual-api-key-here" .env; then
        echo "⚠️  Remember to set your actual OpenAI API key in .env file"
    else
        echo "✅ .env file configured"
    fi
else
    echo "⚠️  Create .env file with your OPENAI_API_KEY"
fi

echo ""
echo "To generate dataset: python Fine_tuning/dataset/generate_dataset.py"
echo "To start GRPO training: python your_grpo_training_script.py" 