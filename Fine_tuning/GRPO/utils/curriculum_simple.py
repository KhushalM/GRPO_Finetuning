"""
Simple Curriculum Learning for GRPO
Progressive System Prompt Reduction: Strong → Medium → Light → Minimal → None

Usage:
    from curriculum_simple import CurriculumLearning
    
    # One-liner setup
    curriculum = CurriculumLearning(epochs=3).setup(dataset, tokenizer)
    
    # Add to trainer
    trainer.add_callback(curriculum.callback)
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from transformers import TrainerCallback
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# CURRICULUM PROMPTS - Compact Definition
# =============================================================================

PROMPTS = {
    1: """You are an expert educator. Approach each question uniquely - sometimes start with a surprising fact, sometimes with a simple question, sometimes with a fundamental principle. Build understanding naturally without following a rigid formula.

CORE PRINCIPLES:
- Vary your opening approach each time
- Use analogies and examples that fit the specific topic
- Build from basics to complexity naturally
- Make your reasoning transparent
- End with an engaging question (but vary how you ask it)

AVOID: Starting every response the same way. Mix up your approach and be conversational. Keep under 256 words.""",

    2: """Explain concepts from first principles using natural, varied approaches. Sometimes use analogies, sometimes direct explanation, sometimes questions. Be flexible and conversational. End with one question.""",

    3: "Build understanding from basics using whatever approach fits best - analogies, examples, or direct reasoning. Vary your style and be natural.",

    4: "Explain from fundamental principles. Use a natural, conversational approach that fits the topic.",

    5: ""  # No prompt — model is free to reason autonomously
}

# Stage timing: More autonomous learning, less rigid prompting
STAGE_TIMING = [0.30, 0.20, 0.20, 0.15, 0.15]  # 30%, 20%, 20%, 15%, 15%

# =============================================================================
# COMPACT CURRICULUM SYSTEM
# =============================================================================

class CurriculumLearning:
    """Ultra-compact curriculum learning system"""
    
    def __init__(self, epochs: int = 3, stages: int = 5):
        self.epochs = epochs
        self.stages = stages
        self.transitions = np.cumsum(STAGE_TIMING) * epochs
        self.dataset = None
        self.callback = None
        self.tokenizer = None
        
        logger.info(f"🎓 Curriculum: {stages} stages over {epochs} epochs")
    
    def get_stage(self, epoch: float) -> int:
        """Get current curriculum stage - UPDATED for more autonomous learning"""
        progress = epoch / self.epochs  # 0.0 to 1.0 progress through training
        
        # 5 stages with timing: 30%, 20%, 20%, 15%, 15%
        if progress <= 0.30:
            return 1  # Strong scaffolding (0-30%)
        elif progress <= 0.50:  # 30% + 20% = 50%
            return 2  # Medium guidance (30-50%)
        elif progress <= 0.70:  # 50% + 20% = 70%
            return 3  # Light hints (50-70%)
        elif progress <= 0.85:  # 70% + 15% = 85%
            return 4  # Minimal prompt (70-85%)
        else:
            return 5  # Autonomous (85-100%)
    
    def get_prompt(self, epoch: float) -> str:
        """Get system prompt for current epoch"""
        return PROMPTS[self.get_stage(epoch)]
    
    def setup(self, dataset, tokenizer):
        """Setup curriculum dataset and callback"""
        self.tokenizer = tokenizer
        self.dataset = CurriculumDataset(dataset, self, tokenizer)
        self.callback = CurriculumCallback(self.dataset, self, tokenizer)
        logger.info(f"✅ Curriculum setup complete: {len(dataset)} examples")
        return self
    
    def get_config(self) -> Dict:
        """Get config for logging"""
        return {
            "curriculum_enabled": True,
            "curriculum_stages": self.stages,
            "curriculum_epochs": self.epochs,
            "stage_timing": "30% → 20% → 20% → 15% → 15%"
        }

# =============================================================================
# CURRICULUM DATASET - Minimal Implementation
# =============================================================================

class CurriculumDataset(torch.utils.data.Dataset):
    """Dataset that adapts prompts based on curriculum stage"""
    
    def __init__(self, base_dataset, curriculum: CurriculumLearning, tokenizer):
        self.base_dataset = base_dataset
        self.curriculum = curriculum
        self.tokenizer = tokenizer
        self.current_epoch = 0
        self.last_stage = 1
    
    def set_epoch(self, epoch: float):
        """Update epoch and log stage transitions"""
        self.current_epoch = epoch
        stage = self.curriculum.get_stage(epoch)
        if stage != self.last_stage:
            stage_names = ["Strong", "Medium", "Light", "Minimal", "None"]
            old_name = stage_names[self.last_stage - 1]
            new_name = stage_names[stage - 1]
            logger.info(f"🎯 Stage {self.last_stage}→{stage}: {old_name}→{new_name}")
            self.last_stage = stage
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """Get item with current curriculum prompt - FIXED for GRPO compatibility"""
        item = self.base_dataset[idx].copy() if hasattr(self.base_dataset[idx], 'copy') else dict(self.base_dataset[idx])
        curriculum_prompt = self.curriculum.get_prompt(self.current_epoch)
        
        # ✅ FIX: GRPO expects "prompt" key with messages format, not "messages" key
        if 'messages' in item:
            messages = item['messages'].copy()
            if curriculum_prompt:  # Add/replace system message
                if messages and messages[0]['role'] == 'system':
                    messages[0]['content'] = curriculum_prompt
                else:
                    messages.insert(0, {'role': 'system', 'content': curriculum_prompt})
            elif messages and messages[0]['role'] == 'system':  # Remove system message for stage 5
                messages = messages[1:]
            
            # ✅ CRITICAL: GRPO expects "prompt" key, not "messages" key
            item['prompt'] = messages
            if 'messages' in item:
                del item['messages']  # Remove to avoid confusion
        
        # ✅ FIX: Convert completion string to messages format if present
        if 'completion' in item and isinstance(item['completion'], str):
            item['completion'] = [{"role": "assistant", "content": item['completion']}]
        
        return item

# =============================================================================
# ENHANCED CURRICULUM CALLBACK WITH WANDB LOGGING
# =============================================================================

class CurriculumCallback(TrainerCallback):
    """Enhanced callback to update curriculum epoch and log to WandB"""
    
    def __init__(self, dataset: CurriculumDataset, curriculum: CurriculumLearning, tokenizer):
        self.dataset = dataset
        self.curriculum = curriculum
        self.tokenizer = tokenizer
        self.last_stage = 1
        self.last_logged_step = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Log initial curriculum setup to WandB"""
        try:
            import wandb
            if wandb.run:
                wandb.log({
                    "curriculum/total_stages": self.curriculum.stages,
                    "curriculum/total_epochs": self.curriculum.epochs,
                    "curriculum/stage_timing_text": "40%→25%→20%→10%→5%"
                }, step=0)
                logger.info("📊 Curriculum setup logged to WandB")
        except ImportError:
            logger.info("WandB not available - skipping remote logging")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        """Update curriculum dataset epoch and log stage changes"""
        self.dataset.set_epoch(state.epoch)
        current_stage = self.curriculum.get_stage(state.epoch)
        current_prompt = self.curriculum.get_prompt(state.epoch)
        stage_names = ["Strong Scaffolding", "Medium Guidance", "Light Hints", "Minimal Prompt", "Autonomous"]
        
        # Log stage transitions
        if current_stage != self.last_stage:
            logger.info(f"🎯 STAGE TRANSITION: {self.last_stage}→{current_stage}")
            logger.info(f"   From: {stage_names[self.last_stage-1]} → To: {stage_names[current_stage-1]}")
            logger.info(f"   Epoch: {state.epoch:.2f} | Step: {state.global_step}")
            
            if current_prompt:
                logger.info(f"   New prompt: {current_prompt[:100]}...")
            else:
                logger.info("   New prompt: None (autonomous mode)")
            
            self.last_stage = current_stage
        
        # Log to WandB
        try:
            import wandb
            if wandb.run:
                # Log curriculum metrics (fixed WandB media type issues)
                wandb.log({
                    "curriculum/current_stage": current_stage,
                    "curriculum/stage_name_text": stage_names[current_stage-1],
                    "curriculum/epoch": state.epoch,
                    "curriculum/prompt_length": len(current_prompt) if current_prompt else 0,
                    "curriculum/has_system_prompt_flag": 1 if current_prompt else 0
                }, step=state.global_step)
                
                # Log system prompt as text (not media) - only on stage transitions
                if current_stage != self.last_stage:
                    prompt_text = current_prompt if current_prompt else "None (Autonomous Mode)"
                    wandb.log({
                        "curriculum/prompt_text_content": prompt_text[:300] + "..." if len(prompt_text) > 300 else prompt_text
                    }, step=state.global_step)
                
                # Log stage transition
                if current_stage != self.last_stage:
                    wandb.log({
                        "curriculum/stage_transition_text": f"Stage {self.last_stage}→{current_stage}",
                        "curriculum/transition_epoch": state.epoch
                    }, step=state.global_step)
        except ImportError:
            pass
        
        logger.info(f"📚 Epoch {state.epoch:.1f} - Stage {current_stage} ({stage_names[current_stage-1]})")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log curriculum info with training metrics"""
        if logs and state.global_step % 25 == 0:  # Log every 25 steps
            current_stage = self.curriculum.get_stage(state.epoch)
            current_prompt = self.curriculum.get_prompt(state.epoch)
            
            try:
                import wandb
                if wandb.run:
                    curriculum_logs = {
                        "curriculum/step_stage": current_stage,
                        "curriculum/step_prompt_active_flag": 1 if current_prompt else 0
                    }
                    wandb.log(curriculum_logs, step=state.global_step)
            except ImportError:
                pass

# =============================================================================
# ENHANCED EVALUATION CALLBACK WITH COMPLETION LOGGING
# =============================================================================

class SimpleEvalCallback(TrainerCallback):
    """Enhanced evaluation callback with completion logging"""
    
    def __init__(self, curriculum: CurriculumLearning, eval_frequency: int = 50):
        self.curriculum = curriculum
        self.eval_frequency = eval_frequency
        self.metrics = SimpleMetrics()
        self.eval_history = []
        self.completion_count = 0
        self.test_questions = [
            "Explain how gravity works",
            "Why do objects fall when dropped?",
            "What causes the tides in the ocean?",
            "How does electricity flow through a wire?"
        ]
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Enhanced evaluation with completion samples"""
        if state.global_step % self.eval_frequency == 0:
            current_stage = self.curriculum.get_stage(state.epoch)
            current_prompt = self.curriculum.get_prompt(state.epoch)
            stage_names = ["Strong", "Medium", "Light", "Minimal", "Auto"]
            
            eval_data = {
                'step': state.global_step,
                'epoch': state.epoch,
                'stage': current_stage,
                'stage_name': stage_names[current_stage-1],
                'has_prompt': bool(current_prompt)
            }
            self.eval_history.append(eval_data)
            
            logger.info(f"🔍 Step {state.global_step}: Stage {current_stage} evaluation logged")
            
            # Generate completion sample for WandB
            if model and self.curriculum.tokenizer:
                self._log_completion_sample(state, model, current_stage, current_prompt, stage_names)
    
    def _log_completion_sample(self, state, model, current_stage, current_prompt, stage_names):
        """Generate and log completion sample to WandB"""
        try:
            import wandb
            if not wandb.run:
                return
            
            # Select test question
            test_question = self.test_questions[self.completion_count % len(self.test_questions)]
            
            # Create messages with current curriculum prompt
            if current_prompt:
                messages = [
                    {"role": "system", "content": current_prompt},
                    {"role": "user", "content": test_question}
                ]
            else:
                messages = [{"role": "user", "content": test_question}]
            
            # Format with tokenizer
            formatted_prompt = self.curriculum.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate sample completion
            inputs = self.curriculum.tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=150, 
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.curriculum.tokenizer.eos_token_id
                )
            
            completion = self.curriculum.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the new completion
            completion = completion[len(formatted_prompt):].strip()
            
            # Log to WandB with proper text formatting
            self.completion_count += 1
            
            # Create formatted completion text
            completion_text = f"""
STEP {state.global_step} | STAGE {current_stage} ({stage_names[current_stage-1]})
{'='*60}

📋 SYSTEM PROMPT:
{current_prompt if current_prompt else "None (Autonomous Mode)"}

❓ QUESTION:
{test_question}

🤖 MODEL RESPONSE:
{completion}

📊 METRICS:
- Epoch: {state.epoch:.2f}
- Prompt Length: {len(current_prompt) if current_prompt else 0} chars
- Response Length: {len(completion)} chars
"""
            
            wandb.log({
                "completions/formatted_sample_text": completion_text,
                "completions/stage": current_stage,
                "completions/has_system_prompt_flag": 1 if current_prompt else 0,
                "completions/prompt_length": len(current_prompt) if current_prompt else 0,
                "completions/response_length": len(completion),
                "completions/question_text": test_question,
                "completions/response_text": completion,
                "completions/system_prompt_used_text": current_prompt if current_prompt else "None"
            }, step=state.global_step)
            
            logger.info(f"📝 Completion sample logged to WandB (Stage {current_stage})")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not generate completion sample: {e}")

# =============================================================================
# SIMPLE EVALUATION METRICS
# =============================================================================

class SimpleMetrics:
    """Compact evaluation metrics for curriculum learning"""
    
    @staticmethod
    def reasoning_score(text: str) -> float:
        """Score reasoning quality (0-1)"""
        indicators = ["because", "therefore", "first", "then", "fundamental", "principle"]
        return min(1.0, sum(indicator in text.lower() for indicator in indicators) / 6)
    
    @staticmethod  
    def independence_score(text: str, stage: int) -> float:
        """Score reasoning independence based on stage"""
        base_score = SimpleMetrics.reasoning_score(text)
        # Bonus for maintaining reasoning without strong prompts
        stage_bonus = {1: 0.0, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.5}
        return min(1.0, base_score + stage_bonus.get(stage, 0.0))
    
    @staticmethod
    def evaluate_response(text: str, stage: int) -> Dict[str, float]:
        """Quick evaluation of response quality"""
        return {
            'reasoning': SimpleMetrics.reasoning_score(text),
            'independence': SimpleMetrics.independence_score(text, stage),
            'overall': (SimpleMetrics.reasoning_score(text) + SimpleMetrics.independence_score(text, stage)) / 2
        }

# =============================================================================
# ONE-LINER SETUP FUNCTIONS
# =============================================================================

def quick_setup(dataset, tokenizer, epochs: int = 3) -> CurriculumLearning:
    """One-liner curriculum setup"""
    return CurriculumLearning(epochs=epochs).setup(dataset, tokenizer)

def add_to_trainer(trainer, curriculum: CurriculumLearning, eval_freq: int = 50):
    """Add curriculum to trainer with optional evaluation"""
    trainer.add_callback(curriculum.callback)
    if eval_freq > 0:
        eval_callback = SimpleEvalCallback(curriculum, eval_freq)
        trainer.add_callback(eval_callback)
        return eval_callback
    return None

# =============================================================================
# COMPACT INTEGRATION EXAMPLE
# =============================================================================

def integrate_curriculum(trainer, dataset, tokenizer, epochs: int = 3, eval_freq: int = 50):
    """Complete integration in one function call"""
    # Setup curriculum
    curriculum = quick_setup(dataset, tokenizer, epochs)
    
    # Add to trainer
    eval_callback = add_to_trainer(trainer, curriculum, eval_freq)
    
    # Update trainer config
    if hasattr(trainer.args, 'report_to') and 'wandb' in trainer.args.report_to:
        try:
            import wandb
            if wandb.run:
                wandb.config.update(curriculum.get_config())
        except ImportError:
            pass
    
    logger.info("🚀 Curriculum learning integrated with WandB logging!")
    logger.info(f"   📊 Dataset: {len(dataset)} examples")
    logger.info(f"   🎯 Stages: 5 (Strong→Medium→Light→Minimal→None)")
    logger.info(f"   ⏱️  Timing: 40%→25%→20%→10%→5%")
    logger.info(f"   🔍 Eval: Every {eval_freq} steps" if eval_freq > 0 else "   🔍 Eval: Disabled")
    logger.info(f"   📝 WandB: System prompts and completions logged")
    
    return curriculum, eval_callback

# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("🎓 Enhanced Curriculum Learning with WandB")
    print("=" * 50)
    
    # Example usage
    print("📝 Basic Usage:")
    print("curriculum = CurriculumLearning(epochs=3).setup(dataset, tokenizer)")
    print("trainer.add_callback(curriculum.callback)  # Now includes WandB logging")
    print()
    
    print("📝 One-liner Integration:")
    print("curriculum, eval_cb = integrate_curriculum(trainer, dataset, tokenizer)")
    print()
    
    print("🎯 Enhanced Features:")
    print("   ✅ 5-stage progressive prompt reduction")
    print("   ✅ Automatic stage transitions")
    print("   ✅ WandB logging with system prompts")
    print("   ✅ Completion samples in WandB")
    print("   ✅ Stage transition tracking")
    print("   ✅ Rich HTML formatting for completions")
    
    # Demo stage progression
    print("\n📊 Stage Progression (3 epochs):")
    curriculum = CurriculumLearning(epochs=3)
    test_epochs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    stage_names = ["Strong", "Medium", "Light", "Minimal", "None"]
    
    for epoch in test_epochs:
        stage = curriculum.get_stage(epoch)
        prompt_len = len(curriculum.get_prompt(epoch))
        print(f"   Epoch {epoch:3.1f} → Stage {stage} ({stage_names[stage-1]:7s}) → Prompt: {prompt_len:3d} chars") 