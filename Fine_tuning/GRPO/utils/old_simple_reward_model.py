"""
Simplified Reward Function
"""
import re
from textstat import flesch_reading_ease

class RewardScorer:
    """
    Rewards explanations on:
      - Step progression
      - Analogy usage
      - Clarity (Flesch reading ease)
      - Conciseness
      - Unique word ratio
      - Adherence to Feynman-style guidelines:
        * Start from fundamentals
        * Use analogies
        * Layer step by step
        * Be transparent
        * End with key insight
        * Single question at end
        * Less than 256 words
    """
    def __init__(
        self,
        w_step=0.25,
        w_analogy=0.20,
        w_clarity=0.20,
        w_length=0.10,
        w_repetition=0.10,
        w_format=0.15
    ):
        total = w_step + w_analogy + w_clarity + w_length + w_repetition + w_format
        assert abs(total - 1.0) < 1e-6, "Weights must sum to 1.0"
        self.w = dict(
            step=w_step,
            analogy=w_analogy,
            clarity=w_clarity,
            length=w_length,
            repetition=w_repetition,
            format=w_format
        )
        # Patterns
        self.step_pattern = re.compile(r'\b(first|then|next|finally)\b', re.I)
        self.analogy_pattern = re.compile(r'\b(like|imagine|picture this)\b', re.I)
        self.fundamentals_pattern = re.compile(
            r'\b(fundamentally|essentially|at its core|from scratch|root cause|building block|foundation)\b', re.I
        )
        self.transparency_pattern = re.compile(
            r'\b(because|therefore|thus|so that|consequently|as a result|hence)\b', re.I
        )
        self.conclusion_pattern = re.compile(
            r'\b(in summary|to sum up|in conclusion|ultimately|this explains|so)\b', re.I
        )

    def clamp(self, x, minimum=0.0, maximum=1.0):
        return max(minimum, min(maximum, x))

    def score(self, text: str) -> float:
        words = text.split()
        wc = len(words)

        # 1) Step progression
        step_score = self.clamp(len(self.step_pattern.findall(text)) / 3)

        # 2) Analogy usage
        ana_score = self.clamp(len(self.analogy_pattern.findall(text)) / 2)

        # 3) Clarity via Flesch
        flesch = flesch_reading_ease(text)
        clarity_score = self.clamp((flesch - 30) / 40)

        # 4) Conciseness (50–200 words ideal)
        length_score = self.clamp((wc - 50) / 150)

        # 5) Unique word ratio
        cleaned = [w.strip('.,!?;:').lower() for w in words if w]
        rep_score = self.clamp(len(set(cleaned)) / len(cleaned)) if cleaned else 1.0

        # 6) Format adherence
        fund_score = 1.0 if self.fundamentals_pattern.search(text) else 0.0
        transp_score = self.clamp(len(self.transparency_pattern.findall(text)) / 1)
        concl_score = 1.0 if self.conclusion_pattern.search(text) else 0.0
        total_q = text.count('?')
        last_q = 1 if text.rstrip().endswith('?') else 0
        internal_q = total_q - last_q
        q_score = last_q * self.clamp(1 - internal_q / 2)
        len_req = 1.0 if wc <= 256 else 0.0
        format_score = (fund_score + transp_score + concl_score + q_score + len_req) / 5

        return (
            self.w['step']       * step_score +
            self.w['analogy']    * ana_score +
            self.w['clarity']    * clarity_score +
            self.w['length']     * length_score +
            self.w['repetition'] * rep_score +
            self.w['format']     * format_score
        )

# Example:
reward_scorer = RewardScorer()
# print(scorer.score(text))
