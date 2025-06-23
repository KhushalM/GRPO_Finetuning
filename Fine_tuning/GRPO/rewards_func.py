import re 
import math
import nltk
from typing import Dict, List, Tuple, Any
from textstat import flesch_reading_ease, flesch_kincaid_grade
from collections import Counter
import openai
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import random

try:
    nltk.data.find("tokenizer/punkt")
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("punkt")
    nltk.download("stopwords")

def reward_opening_hook(response: str) -> float:
    """
    Reward randomly between 0.5 and 1.0
    """
    return random.uniform(0.5, 1.0)

class FirstPrinciplesRewardFunction:
     """
     Comprehensive reward function for evaluating first principles explanations
     following the Feynman method: simple analogies, step-by-step reasoning, 
     engaging storytelling, and fundamental understanding.
    """
     def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct"):
        self.weights = {
             'analogy_quality': 0.20,
             'step_by_step_reasoning': 0.15,
             'fundamental_understanding': 0.20,
             'overall_engagement': 0.15,
             'clarity': 0.15,
             'completeness': 0.10,
             'avoid_jargon': 0.05
        }
        
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model = "cardiffnlp/twitter-roberta-base-sentiment-latest",
                device = 0 if torch.cuda.is_available() else -1
            )
        except Exception as e:
             self.sentiment_analyzer = None
             print(f"Warning: Could not load sentimetn analyzer: {e}")

        self.jargon_terms = {
           'utilize', 'paradigm', 'synergy', 'leverage', 'optimize', 'streamline',
            'methodology', 'framework', 'infrastructure', 'scalable', 'robust',
            'innovative', 'cutting-edge', 'state-of-the-art', 'holistic', 'comprehensive'
        }
        
        self.first_principles_indicators = {
            'fundamental_starters': [
                'imagine', 'think of', 'picture', 'let\'s start with', 'at its core',
                'fundamentally', 'basically', 'essentially', 'from the beginning',
                'the basic idea', 'the foundation'
            ],
            'analogy_patterns': [
                'like', 'similar', 'imagine', 'think of', 'picture', 'as if',
                'it\'s like when', 'just like', 'similar to', 'comparable to'
            ],
            'step_indicators': [
                'first', 'second', 'third', 'next', 'then', 'after that',
                'step by step', 'one by one', 'gradually', 'building up'
            ],
            'engagement_patterns': [
                'does this', 'do you see', 'can you picture', 'have you noticed',
                'does this help', 'make sense', 'clear now', 'understand how'
            ]
        }

     def evaluate_analogy_quality(self, response: str) -> float:
        """
        Evaluate the quality of the presence of analogies in the response.
        Reward concrete, relatable, and easy to understand analogies.
        """

        score =0.0
        response_lower = response.lower()
        
        abalogy_count = 0

        for pattern in self.first_principles_indicators['analogy_patterns']:
             analogy_count += len(re.findall(rf'\b{pattern}\b', response_lower))

        if analogy_count > 0:
             score += 0.3
        
        concrete_examples = [
             'ball', 'car', 'house', 'water', 'air', 'food', 'game', 'toy',
            'bicycle', 'seesaw', 'playground', 'kitchen', 'garden', 'road',
            'bridge', 'ladder', 'puzzle', 'painting', 'story', 'movie'
        ]

        concrete_count = sum(1 for word in response_lower.split() if word in concrete_examples)

        score += min(0.4, concrete_count * 0.1)

        sensory_words = [
             'see', 'feel', 'hear', 'touch', 'taste', 'smell', 'warm', 'cold',
            'bright', 'dark', 'smooth', 'rough', 'loud', 'quiet'
        ]

        sensory_count = sum(1 for word in response_lower.split() if word in sensory_words)

        score += min(0.3, sensory_count * 0.05)

        return min(1.0, score)


     def evaluate_step_by_step_reasoning(self, response: str) -> float:
        """
        Evaluate if the explanation follows a logical, step-by-step progression
        """
        score = 0.0
        response_lower = response.lower()
        
        # Check for step indicators
        step_indicators = self.first_principles_indicators['step_indicators']
        step_count = sum(1 for indicator in step_indicators 
                        if indicator in response_lower)
        
        # Reward presence of step indicators (0.4 points)
        score += min(0.4, step_count * 0.1)
        
        # Check for logical connectors
        connectors = [
            'because', 'so', 'therefore', 'as a result', 'this means',
            'which leads to', 'causing', 'resulting in', 'this is why'
        ]
        connector_count = sum(1 for connector in connectors 
                            if connector in response_lower)
        
        # Reward logical flow (0.3 points)
        score += min(0.3, connector_count * 0.1)
        
        # Check for building complexity (simple to complex)
        sentences = nltk.sent_tokenize(response)
        if len(sentences) >= 3:
            # Simple heuristic: later sentences should build on earlier ones
            early_sentence_length = sum(len(s.split()) for s in sentences[:len(sentences)//2])
            later_sentence_length = sum(len(s.split()) for s in sentences[len(sentences)//2:])
            
            if later_sentence_length > early_sentence_length:
                score += 0.3  # Reward building complexity
        
        return min(1.0, score)


     def evaluate_fundamental_concepts(self, response: str) -> float:
        """
        Evaluate if the explanation addresses fundamental concepts rather than
        just surface-level descriptions
        """
        score = 0.0
        response_lower = response.lower()
        
        # Check for fundamental starters
        fundamental_starters = self.first_principles_indicators['fundamental_starters']
        starter_count = sum(1 for starter in fundamental_starters 
                          if starter in response_lower)
        
        # Reward fundamental approach (0.4 points)
        score += min(0.4, starter_count * 0.2)
        
        # Check for "why" reasoning (getting to root causes)
        why_patterns = ['why', 'reason', 'cause', 'because', 'due to', 'leads to']
        why_count = sum(1 for pattern in why_patterns 
                       if pattern in response_lower)
        
        # Reward causal reasoning (0.3 points)
        score += min(0.3, why_count * 0.05)
        
        # Check for first principle building blocks
        building_blocks = [
            'basic', 'fundamental', 'core', 'essential', 'underlying',
            'foundation', 'principle', 'rule', 'law', 'truth'
        ]
        building_count = sum(1 for block in building_blocks 
                           if block in response_lower)
        
        # Reward first principles approach (0.3 points)
        score += min(0.3, building_count * 0.1)
        
        return min(1.0, score)

     def evaluate_engagement(self, response: str) -> float:
        """
        Evaluate how engaging and interactive the explanation is
        """
        score = 0.0
        response_lower = response.lower()
        
        # Check for engagement patterns (questions, direct address)
        engagement_patterns = self.first_principles_indicators['engagement_patterns']
        engagement_count = sum(1 for pattern in engagement_patterns 
                             if pattern in response_lower)
        
         # Reward engagement (0.4 points)
        score += min(0.4, engagement_count * 0.1)
        
        # Check for questions to the reader
        question_count = response.count('?')
        score += min(0.3, question_count * 0.1)
        
        # Use sentiment analysis for positivity (engaging tone)
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer(response[:512])  # Truncate for API limits
                if sentiment[0]['label'] == 'POSITIVE':
                    score += 0.3
            except:
                pass  # Skip if sentiment analysis fails
        
        return min(1.0, score)

     def evaluate_clarity(self, response: str) -> float:
        """
        Evaluate clarity using readability metrics and sentence structure
        """
        score = 0.0
        
        # Flesch Reading Ease (higher is better, 60-70 is ideal for general audience)
        try:
            flesch_score = flesch_reading_ease(response)
            if flesch_score >= 60:
                score += 0.4
            elif flesch_score >= 50:
                score += 0.3
            elif flesch_score >= 40:
                score += 0.2
            else:
                score += 0.1
        except:
            score += 0.2  # Default if calculation fails
        
        # Average sentence length (12-18 words is ideal)
        sentences = nltk.sent_tokenize(response)
        if sentences:
            avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
            if 10 <= avg_sentence_length <= 20:
                score += 0.3
            elif 8 <= avg_sentence_length <= 25:
                score += 0.2
            else:
                score += 0.1
        
        # Reward simple, clear language
        simple_words = ['the', 'a', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for']
        word_count = len(response.split())
        simple_word_ratio = sum(1 for word in response.lower().split() 
                               if word in simple_words) / max(word_count, 1)
        
        if simple_word_ratio >= 0.3:
            score += 0.3
        
        return min(1.0, score)

     def evaluate_completeness(self, response: str) -> float:
        """
        Evaluate if the explanation is complete and addresses the question fully
        """
        score = 0.0
        
        # Check response length (should be substantial but not excessive)
        word_count = len(response.split())
        if 50 <= word_count <= 300:
            score += 0.5
        elif 30 <= word_count <= 400:
            score += 0.3
        else:
            score += 0.1
        
        # Check for conclusion or summary
        conclusion_indicators = [
            'so', 'therefore', 'in summary', 'to summarize', 'overall',
            'does this help', 'now you can see', 'this explains'
        ]
        
        response_lower = response.lower()
        for indicator in conclusion_indicators:
            if indicator in response_lower:
                score += 0.3
                break
        
        # Check for examples or applications
        example_indicators = ['example', 'for instance', 'like when', 'such as']
        example_count = sum(1 for indicator in example_indicators 
                          if indicator in response_lower)
        
        score += min(0.2, example_count * 0.1)
        
        return min(1.0, score)

     def evaluate_jargon_avoidance(self, response: str) -> float:
        """
        Penalize use of jargon and reward simple, accessible language
        """
        words = response.lower().split()
        jargon_count = sum(1 for word in words if word in self.jargon_terms)
        total_words = len(words)
        
        if total_words == 0:
            return 1.0
        
        jargon_ratio = jargon_count / total_words

        # Penalize jargon usage
        if jargon_ratio == 0:
            return 1.0
        elif jargon_ratio <= 0.02:  # Less than 2% jargon
            return 0.8
        elif jargon_ratio <= 0.05:  # Less than 5% jargon
            return 0.6
        else:
            return 0.3

     def compute_reward(self, response: str, context: str = None) -> Dict[str, float]:
        """
        Compute the overall reward score for a first principles explanation
        """
        scores = {
            'analogy_quality': self.evaluate_analogy_quality(response),
            'step_by_step': self.evaluate_step_by_step_reasoning(response),
            'fundamental_concepts': self.evaluate_fundamental_concepts(response),
            'engagement': self.evaluate_engagement(response),
            'clarity': self.evaluate_clarity(response),
            'completeness': self.evaluate_completeness(response),
            'avoid_jargon': self.evaluate_jargon_avoidance(response)
        }
        
        # Calculate weighted total
        total_score = sum(scores[key] * self.weights[key] for key in scores.keys())
        
        scores['total'] = total_score
        scores['normalized'] = min(1.0, max(0.0, total_score))
        
        return scores

# Initialize the reward function
reward_evaluator = FirstPrinciplesRewardFunction()

def reward_opening_hook(response: str, context: str = None) -> float:
    """
    Main reward function for GRPO training.
    Returns a score between 0.0 and 1.0 based on first principles explanation quality.
    """
    scores = reward_evaluator.compute_reward(response, context)
    return scores['normalized']

def detailed_reward_analysis(response: str, context: str = None) -> Dict[str, Any]:
    """
    Returns detailed breakdown of reward components for analysis
    """
    return reward_evaluator.compute_reward(response, context)

def reward_with_feedback(response: str, context: str = None) -> Tuple[float, str]:
    """
    Returns reward score and human-readable feedback
    """
    scores = reward_evaluator.compute_reward(response, context)
    
    feedback_parts = []
    
    # Provide specific feedback based on scores
    if scores['analogy_quality'] < 0.5:
        feedback_parts.append("Consider adding more concrete analogies or examples to make the concept relatable.")
    
    if scores['step_by_step'] < 0.5:
        feedback_parts.append("Try breaking down the explanation into clearer, sequential steps.")
    
    if scores['fundamental_concepts'] < 0.5:
        feedback_parts.append("Focus more on the fundamental 'why' and underlying principles.")
    
    if scores['engagement'] < 0.5:
        feedback_parts.append("Make the explanation more engaging with questions or direct address to the reader.")
    
    if scores['clarity'] < 0.5:
        feedback_parts.append("Simplify the language and sentence structure for better clarity.")
    
    if scores['completeness'] < 0.5:
        feedback_parts.append("Provide a more complete explanation with examples and conclusion.")
    
    if scores['avoid_jargon'] < 0.7:
        feedback_parts.append("Avoid technical jargon and use simpler, more accessible language.")

    feedback = " ".join(feedback_parts) if feedback_parts else "Great first principles explanation!"
    
    return scores['normalized'], feedback

# Additional utility functions for GRPO training

def batch_reward_computation(responses: List[str], contexts: List[str] = None) -> List[float]:
    """
    Compute rewards for a batch of responses efficiently
    """
    if contexts is None:
        contexts = [None] * len(responses)
    
    return [reward_opening_hook(response, context) 
            for response, context in zip(responses, contexts)]

def reward_calibration_test():
    """
    Test the reward function with sample responses to ensure proper calibration
    """
    test_cases = [
        # High quality first principles explanation
        """Okay, imagine you have a stretched rubber sheet and you place a heavy ball in the middle. The sheet bends downwards, right? Now, if you roll a smaller ball nearby, it will start rolling toward the heavier ball because of the dip. This is a simple way to picture how gravity works. Gravity is like the Earth making a 'dip' in space that pulls things toward it. When you let go of an object, it falls because the Earth is pulling it toward its center, similar to how the heavy ball makes the rubber sheet dip. Does this help you see why objects fall when dropped?""",
        
        # Medium quality explanation
        """Gravity is a fundamental force that attracts objects with mass toward each other. The Earth has a very large mass, so it pulls objects toward its center. When you drop something, gravity causes it to accelerate downward at 9.8 meters per second squared.""",
        
        # Low quality jargon-heavy explanation
        """The gravitational force is a manifestation of the curvature of spacetime as described by Einstein's field equations. Objects follow geodesics in this curved spacetime, which manifests as gravitational attraction in classical mechanics."""
    ]
    
    for i, test_case in enumerate(test_cases):
        score, feedback = reward_with_feedback(test_case)
        print(f"Test Case {i+1}:")
        print(f"Score: {score:.3f}")
        print(f"Feedback: {feedback}")
        print("-" * 50)

if __name__ == "__main__":
    reward_calibration_test()