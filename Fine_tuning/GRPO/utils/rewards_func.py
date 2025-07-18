"""
Reward function for GRPO / RLHF
• Seven Feynman-style axes → single 0-to-1 reward
• CUDA-aware, sentiment batched on GPU
• W&B logs three sample completions per call (no step mismatch)
• Enhanced with negative rewarding for undesirable behaviors
"""

# -- Imports & env ─────────────────────────────────────────────────────────────────
import os, re
from typing import List, Dict

import torch
import nltk
from packaging import version
from transformers import pipeline
from textstat import flesch_reading_ease
import wandb

DEVICE = 0 if torch.cuda.is_available() else -1          # -1 → CPU

# -- One-time W&B init (safe if already initialised) --
if os.getenv("WANDB_DISABLED", "false").lower() != "true" and wandb.run is None:
    wandb.init(project="grpo-first-principles", reinit=True)

# -- Secure NLTK data (punkt_tab ≥ 3.8.2) --
if version.parse(nltk.__version__) >= version.parse("3.8.2"):
    nltk.download("punkt_tab", quiet=True)
    SENT_TOKENIZE = nltk.data.load("tokenizers/punkt_tab/english.pickle").tokenize
else:
    nltk.download("punkt", quiet=True)
    SENT_TOKENIZE = nltk.sent_tokenize

# -- Sentiment pipeline (batched) --
try:
    SENT_PIPE = pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        device=DEVICE,
        torch_dtype=torch.float16 if DEVICE >= 0 else None,
        batch_size=32,
    )
except Exception:
    SENT_PIPE = None  # sentiment optional

# -- Static vocabularies → CUDA tensors for bucketised lookup --
_CONCRETE = [
    "ball", "car", "house", "water", "air", "food", "game", "toy", "bicycle",
    "apple", "book", "chair", "door", "phone", "street", "bridge", "cup",
    "sand", "river", "tree", "mountain", "train", "clock", "computer", "lamp",
]

# Words that evoke a direct sensory experience
_SENSORY = [
    # vision
    "see", "look", "bright", "dark", "color", "shine", "glow",
    # touch / temperature
    "feel", "touch", "smooth", "rough", "soft", "hard", "warm", "cold",
    # hearing
    "hear", "sound", "loud", "quiet", "crackle", "whisper",
    # taste / smell
    "taste", "smell", "sweet", "bitter", "sour", "salty", "fresh", "fragrant",
]
_JARGON    = [
    'utilize','paradigm','synergy','leverage','optimize','streamline','methodology',
    'framework','infrastructure','scalable','robust','innovative','state-of-the-art'
]
def _vocab_tensor(words):      # hashed & sorted for bucketize
    return torch.tensor(sorted(hash(w) for w in words), device=DEVICE)
CONCRETE_T, SENSORY_T, JARGON_T = map(_vocab_tensor, (_CONCRETE, _SENSORY, _JARGON))

def _batch_hash(tokens: List[List[str]]) -> torch.Tensor:
    """Pad to equal length and return int64 [B,L] tensor of word hashes on DEVICE."""
    L = max(len(t) for t in tokens)
    mat = torch.full((len(tokens), L), 0, dtype=torch.int64, device=DEVICE)
    for i, tok in enumerate(tokens):
        mat[i, :len(tok)] = torch.tensor([hash(w) for w in tok], device=DEVICE)
    return mat

class FirstPrinciplesRewardV3:
    def __init__(self, device="cuda", max_batch_size=64, enable_caching=True):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_batch_size = max_batch_size
        self.enable_caching = enable_caching
        self._pattern_cache = {} if enable_caching else None

        # ✅ Enhanced weights with negative components
        self.WEIGHTS = {
            # Positive components (sum to ~0.9)
            "analogy": 0.25,           # Most important for Feynman style
            "step": 0.20,              # Crucial for good explanations
            "fundamentals": 0.18,      # Keep high
            "engagement": 0.12,        # Good for interaction
            "clarity": 0.08,           # Readability matters
            "completeness": 0.05,      # Some length consideration
            "no_jargon": 0.02,         # Minimal - absence less important
            
            # ✅ NEW: Negative components (sum to ~0.1 penalty potential)
            "no_self_talk": 0.03,      # Penalize self-conversations
            "no_repetition": 0.03,     # Penalize excessive repetition
            "coherence": 0.02,         # Penalize rambling/off-topic
            "structure": 0.02,         # Penalize poor organization
        }

        self._init_vocabularies()
        self._compile_patterns()

    def _init_vocabularies(self):
        """Initialize enhanced vocabularies with semantic grouping"""
        # Concrete objects (expanded and categorized)
        self.CONCRETE_PHYSICAL = [
            "ball", "car", "house", "water", "air", "food", "bicycle", "apple", "book",
            "chair", "door", "phone", "bridge", "cup", "sand", "river", "tree", "clock",
            "sheet", "rubber", "bowling", "marble", "fabric", "space", "earth", "moon",
            "pen", "slope", "dip", "curve", "surface", "weight"
        ]
        
        self.CONCRETE_ABSTRACT = [
            "game", "story", "picture", "music", "dance", "recipe", "map", "puzzle",
            "path", "journey", "process", "method", "system"
        ]
        
        # Sensory experience words (enhanced)
        self.SENSORY_VISUAL = ["see", "look", "bright", "dark", "color", "shine", "glow", "sparkle", "watch", "observe"]
        self.SENSORY_TACTILE = ["feel", "touch", "smooth", "rough", "soft", "hard", "warm", "cold", "heavy", "light"]
        self.SENSORY_AUDITORY = ["hear", "sound", "loud", "quiet", "whisper", "crackle", "ring", "noise"]
        self.SENSORY_OTHER = ["taste", "smell", "sweet", "bitter", "fresh", "fragrant"]
        
        # Convert to GPU tensors with improved hashing
        self.vocab_tensors = {}
        for name, vocab in {
            "concrete_physical": self.CONCRETE_PHYSICAL,
            "concrete_abstract": self.CONCRETE_ABSTRACT,
            "sensory_visual": self.SENSORY_VISUAL,
            "sensory_tactile": self.SENSORY_TACTILE,
            "sensory_auditory": self.SENSORY_AUDITORY,
            "sensory_other": self.SENSORY_OTHER,
        }.items():
            # Use stable hash for consistent results
            hashes = [hash(w) & 0x7FFFFFFF for w in vocab]  # Ensure positive
            self.vocab_tensors[name] = torch.tensor(sorted(hashes), device=self.device)

    def _compile_patterns(self):
        """Compile enhanced regex patterns with better coverage"""
        self.patterns = {
            "analogy_strong": re.compile(
                r'\b(like|similar to|imagine|as if|just like|comparable to|think of it as|'
                r'picture this|it\'s like|reminds me of|analogous to|sort of like)\b', re.I
            ),
            "analogy_weak": re.compile(
                r'\b(kind of|sort of|similar|resemble|compare|akin to)\b', re.I
            ),
            "step_indicators": re.compile(
                r'\b(first|second|third|next|then|after that|step by step|'
                r'initially|subsequently|finally|to begin|to start|now|when|following)\b', re.I
            ),
            "causal_connectors": re.compile(
                r'\b(because|therefore|as a result|this is why|which leads to|'
                r'consequently|thus|hence|so that|due to|since|leads to)\b', re.I
            ),
            "fundamental_phrases": re.compile(
                r'\b(at its core|at the core|fundamentally|basically|essentially|'
                r'from scratch|from the ground up|first principles?|'
                r'root cause|underlying|the essence|building block|foundation|'
                r'the basic idea|in essence|the nature of)\b', re.I
            ),
            "engagement_direct": re.compile(
                r'\b(does this|does that|do you see|can you see|can you picture|can you imagine|'
                r'have you noticed|have you ever|could you picture|picture this|'
                r'make sense|sound good|is that clear|is this clear|see how|'
                r'notice how|feel how|think about|consider this)\b', re.I
            ),
            "engagement_questions": re.compile(r'\?', re.I),
            "conclusion_indicators": re.compile(
                r'\b(so|therefore|in summary|overall|this explains|'
                r'to sum up|in conclusion|ultimately|does this help)\b', re.I
            ),
            "example_phrases": re.compile(
                r'\b(for example|for instance|such as|like when|'
                r'consider|take|let\'s say|imagine if)\b', re.I
            ),
            
            # ✅ NEW: Negative patterns
            "self_conversation": re.compile(
                r'(Sure,?\s+let\'s|Yes,?\s+(it|this|that)\s+(helps?|makes?)|'
                r'Can you\s+.*?\?\s+Sure|\?\s+(Yes|Sure|Absolutely)|'
                r'User:|Human:|Assistant:|AI:|You:|Me:)', re.I
            ),
            "repetitive_phrases": re.compile(
                r'(\b\w+\b)(?:\s+\w+){0,3}\s+\1', re.I  # Detects word repetition within short spans
            ),
            "rambling_indicators": re.compile(
                r'\b(um|uh|well|anyway|so yeah|you know|i mean|like i said|'
                r'as i mentioned|going back to|but anyway)\b', re.I
            ),
            "poor_transitions": re.compile(
                r'\b(and and|but but|so so|then then|also also|'
                r'additionally additionally|furthermore furthermore)\b', re.I
            ),
        }

    # ✅ NEW: Negative reward detection methods
    def _detect_self_conversation(self, texts: List[str]) -> torch.Tensor:
        """Detect if the model is having a conversation with itself"""
        batch_size = len(texts)
        penalties = torch.zeros(batch_size, device=self.device)
        
        for i, text in enumerate(texts):
            penalty = 0.0
            
            # Check for conversation patterns
            if self.patterns["self_conversation"].search(text):
                penalty += 0.5  # Major penalty for self-talk
            
            # Check for multiple questions followed by answers
            questions = text.count('?')
            if questions > 2:  # More than 2 questions suggests dialogue
                penalty += min(0.3, (questions - 2) * 0.1)
            
            # Check for role indicators
            role_indicators = ['user:', 'human:', 'assistant:', 'ai:', 'you:', 'me:']
            if any(indicator in text.lower() for indicator in role_indicators):
                penalty += 0.4
            
            penalties[i] = min(1.0, penalty)  # Cap at 1.0
        
        return 1.0 - penalties  # Convert penalty to reward (0 = bad, 1 = good)

    def _detect_repetition(self, texts: List[str]) -> torch.Tensor:
        """Detect excessive repetition in explanations"""
        batch_size = len(texts)
        scores = torch.zeros(batch_size, device=self.device)
        
        for i, text in enumerate(texts):
            words = text.lower().split()
            if len(words) < 10:
                scores[i] = 1.0  # Short texts get benefit of doubt
                continue
            
            # Count word repetitions
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Only count meaningful words
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Calculate repetition score
            total_meaningful_words = sum(1 for w in words if len(w) > 3)
            if total_meaningful_words == 0:
                scores[i] = 1.0
                continue
            
            repeated_words = sum(max(0, count - 2) for count in word_counts.values())
            repetition_ratio = repeated_words / total_meaningful_words
            
            # Check for phrase repetition using patterns
            phrase_repetitions = len(self.patterns["repetitive_phrases"].findall(text))
            phrase_penalty = min(0.3, phrase_repetitions * 0.1)
            
            # Combined score
            repetition_penalty = repetition_ratio + phrase_penalty
            scores[i] = max(0.0, 1.0 - repetition_penalty * 2)  # Amplify penalty
        
        return scores

    def _detect_poor_coherence(self, texts: List[str]) -> torch.Tensor:
        """Detect rambling or incoherent explanations"""
        batch_size = len(texts)
        scores = torch.zeros(batch_size, device=self.device)
        
        for i, text in enumerate(texts):
            penalty = 0.0
            
            # Rambling indicators
            rambling_count = len(self.patterns["rambling_indicators"].findall(text))
            penalty += min(0.4, rambling_count * 0.1)
            
            # Poor transitions
            poor_transitions = len(self.patterns["poor_transitions"].findall(text))
            penalty += min(0.3, poor_transitions * 0.15)
            
            # Sentence length variance (very long or very short sentences suggest poor structure)
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            if sentences:
                lengths = [len(s.split()) for s in sentences]
                if lengths:
                    avg_length = sum(lengths) / len(lengths)
                    # Penalize if average sentence length is too high or if there's extreme variance
                    if avg_length > 35:  # Very long sentences
                        penalty += 0.2
                    elif avg_length < 5:  # Very short sentences
                        penalty += 0.1
                    
                    # Check for extreme length variance
                    if len(lengths) > 1:
                        max_len, min_len = max(lengths), min(lengths)
                        if max_len > min_len * 5:  # High variance suggests poor structure
                            penalty += 0.1
            
            scores[i] = max(0.0, 1.0 - penalty)
        
        return scores

    def _detect_poor_structure(self, texts: List[str]) -> torch.Tensor:
        """Detect poor structural organization"""
        batch_size = len(texts)
        scores = torch.zeros(batch_size, device=self.device)
        
        for i, text in enumerate(texts):
            score = 1.0  # Start with perfect score
            
            # Check for basic structural elements
            has_intro = any(phrase in text.lower()[:100] for phrase in 
                           ['let me explain', 'to understand', 'imagine', 'think of', 'consider'])
            has_conclusion = any(phrase in text.lower()[-100:] for phrase in 
                               ['so', 'therefore', 'this explains', 'does this help', 'make sense'])
            
            # Penalty for missing structure
            if not has_intro:
                score -= 0.2
            if not has_conclusion:
                score -= 0.2
            
            # Check for paragraph breaks (good structure)
            paragraphs = text.split('\n\n')
            if len(paragraphs) > 1 and len(text.split()) > 100:
                score += 0.1  # Bonus for paragraph structure
            
            # Penalty for wall of text (no punctuation variety)
            punctuation_variety = len(set(re.findall(r'[.!?;:,]', text)))
            if punctuation_variety < 2 and len(text.split()) > 50:
                score -= 0.1
            
            scores[i] = max(0.0, min(1.0, score))
        
        return scores
    
    def batch_score_optimized(self, texts: List[str], return_breakdown=False):
        """Optimized batch scoring with memory management"""
        if not texts:
            return []
        
        all_scores = []
        all_breakdowns = [] if return_breakdown else None

        for i in range(0, len(texts), self.max_batch_size):
            batch = texts[i:i + self.max_batch_size]
            scores, breakdowns = self._process_batch(batch, return_breakdown)
            all_scores.extend(scores)
            if return_breakdown:
                all_breakdowns.extend(breakdowns)
        
        return (all_scores, all_breakdowns) if return_breakdown else all_scores

    def _process_batch(self, texts: List[str], return_breakdown=False):
        """Process a single batch with enhanced scoring including negative rewards"""
        batch_size = len(texts)
        texts_lower = [t.lower() for t in texts]

        # Tokenize and create hash matrix
        tokens = [t.split() for t in texts_lower]
        max_len = max(len(t) for t in tokens) if tokens else 0
        
        # Create padded hash matrix on GPU
        hash_matrix = torch.zeros((batch_size, max_len), dtype=torch.int64, device=self.device)
        for i, tok_list in enumerate(tokens):
            if tok_list:
                hashes = [hash(w) & 0x7FFFFFFF for w in tok_list]
                hash_matrix[i, :len(hashes)] = torch.tensor(hashes, device=self.device)
        
        # Compute all scores (positive and negative)
        scores = self._compute_enhanced_scores(texts, texts_lower, hash_matrix, tokens)
        
        # ✅ UPDATED: Weighted combination with TRUE negative penalties
        total_scores = torch.zeros(batch_size, device=self.device)
        
        # Add positive components
        positive_components = ["analogy", "step", "fundamentals", "engagement", "clarity", "completeness", "no_jargon"]
        for component in positive_components:
            if component in scores:
                total_scores += self.WEIGHTS[component] * scores[component]
        
        # Subtract negative penalties (convert good scores to penalties)
        negative_components = ["no_self_talk", "no_repetition", "coherence", "structure"]
        for component in negative_components:
            if component in scores:
                penalty = 1.0 - scores[component]  # Convert 1.0=good to 0.0=no_penalty
                total_scores -= self.WEIGHTS[component] * penalty
        
        # Use tanh to allow negative scores while keeping reasonable range
        final_scores = torch.tanh(total_scores).tolist()  # Range: -1.0 to +1.0
        
        if return_breakdown:
            breakdowns = []
            for i in range(batch_size):
                breakdown = {k: v[i].item() if isinstance(v, torch.Tensor) else v[i] for k, v in scores.items()}
                breakdown["reward"] = final_scores[i]
                breakdowns.append(breakdown)
            return final_scores, breakdowns
        
        return final_scores, None
        
    def _compute_enhanced_scores(self, texts, texts_lower, hash_matrix, tokens):
        """Compute enhanced scoring with FIXED step-by-step detection and negative rewards"""
        batch_size = len(texts)
        
        # ✅ FIXED: Pattern matching with COUNT instead of boolean
        pattern_counts = {}
        for name, pattern in self.patterns.items():
            counts = torch.zeros(batch_size, device=self.device)
            for i, text_lower in enumerate(texts_lower):
                matches = len(pattern.findall(text_lower))  # COUNT matches, not just boolean
                counts[i] = matches
            pattern_counts[name] = counts
        
        # Vocabulary matching on GPU (unchanged)
        vocab_counts = {}
        for name, vocab_tensor in self.vocab_tensors.items():
            matches = torch.isin(hash_matrix, vocab_tensor)
            counts = torch.sum(matches, dim=1).float()
            vocab_counts[name] = counts
        
        # ✅ POSITIVE COMPONENTS (unchanged)
        step_scores = (
            0.6 * torch.clamp(pattern_counts["step_indicators"] / 3.0, max=1.0) +
            0.3 * torch.clamp(pattern_counts["causal_connectors"] / 2.0, max=1.0) +
            0.1 * self._compute_progression_score(texts)
        )
        
        analogy_scores = (
            0.5 * torch.clamp(pattern_counts["analogy_strong"] / 2.0, max=1.0) +
            0.2 * torch.clamp(pattern_counts["analogy_weak"] / 3.0, max=1.0) +
            0.2 * torch.clamp(vocab_counts["concrete_physical"] / 6.0, max=1.0) +
            0.05 * torch.clamp(vocab_counts["sensory_visual"] / 3.0, max=1.0) +
            0.05 * torch.clamp(vocab_counts["sensory_tactile"] / 2.0, max=1.0)
        )
        
        fundamental_scores = (
            0.7 * torch.clamp(pattern_counts["fundamental_phrases"] / 2.0, max=1.0) +
            0.2 * self._compute_depth_score(texts_lower) +
            0.1 * torch.clamp(vocab_counts["concrete_physical"] / 12.0, max=1.0)
        )
        
        engagement_scores = (
            0.4 * torch.clamp(pattern_counts["engagement_direct"] / 2.0, max=1.0) +
            0.3 * torch.clamp(pattern_counts["engagement_questions"] / 2.0, max=1.0) +
            0.3 * self._compute_interactive_score(texts)
        )
        
        clarity_scores = self._compute_clarity_score(texts)
        completeness_scores = self._compute_completeness_score(texts)
        jargon_scores = self._compute_jargon_score(texts_lower)
        
        # ✅ NEW: NEGATIVE COMPONENTS
        no_self_talk_scores = self._detect_self_conversation(texts)
        no_repetition_scores = self._detect_repetition(texts)
        coherence_scores = self._detect_poor_coherence(texts)
        structure_scores = self._detect_poor_structure(texts)
        
        return {
            # Positive components
            "analogy": analogy_scores,
            "step": step_scores,
            "fundamentals": fundamental_scores,
            "engagement": engagement_scores,
            "clarity": clarity_scores,
            "completeness": completeness_scores,
            "no_jargon": jargon_scores,
            
            # Negative components (higher score = less penalty)
            "no_self_talk": no_self_talk_scores,
            "no_repetition": no_repetition_scores,
            "coherence": coherence_scores,
            "structure": structure_scores,
        }

    def _compute_progression_score(self, texts: List[str]) -> torch.Tensor:
        """Enhanced progression scoring to better detect step-by-step explanations"""
        batch_size = len(texts)
        scores = torch.zeros(batch_size, device=self.device)
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            
            # Count explicit sequence indicators
            sequence_words = ['first', 'second', 'third', 'fourth', 'fifth', 
                             'next', 'then', 'after that', 'finally', 'lastly']
            sequence_count = sum(1 for word in sequence_words if word in text_lower)
            
            # Count transition words
            transitions = ['following', 'subsequently', 'now', 'when', 'after']
            transition_count = sum(1 for word in transitions if word in text_lower)
            
            total_indicators = sequence_count + transition_count
            
            if total_indicators >= 5:      # Excellent progression
                scores[i] = 1.0
            elif total_indicators >= 3:    # Good progression  
                scores[i] = 0.8
            elif total_indicators >= 2:    # Some progression
                scores[i] = 0.6
            elif total_indicators >= 1:    # Minimal progression
                scores[i] = 0.4
            else:
                scores[i] = 0.2
        
        return scores

    def _compute_depth_score(self, texts_lower: List[str]) -> torch.Tensor:
        """Compute depth score for fundamental understanding (more generous)"""
        batch_size = len(texts_lower)
        scores = torch.zeros(batch_size, device=self.device)
        
        depth_words = ['why', 'because', 'reason', 'cause', 'underlying', 'fundamental',
                      'how', 'what', 'explains', 'mechanism', 'process', 'nature']
        
        for i, text in enumerate(texts_lower):
            depth_count = sum(1 for word in depth_words if word in text)
            scores[i] = min(1.0, depth_count / 3.0)  # More generous normalization
        
        return scores

    def _compute_interactive_score(self, texts: List[str]) -> torch.Tensor:
        """Compute interactive engagement score with self-talk detection"""
        batch_size = len(texts)
        scores = torch.zeros(batch_size, device=self.device)
        
        interactive_patterns = [
            r'\b(do you|can you|have you|did you|will you|would you)\b',
            r'\b(imagine|picture this|think about|consider|notice)\b',
            r'\b(let\'s|we can|we should|we need)\b',
            r'\b(see how|feel how|watch|observe)\b',
            r'\b(does this|is this|makes sense|clear|understand)\b'
        ]
        
        for i, text in enumerate(texts):
            text_lower = text.lower()
            score = 0.0
            
            # Count questions (more generous)
            question_count = text.count('?')
            score += min(0.5, question_count * 0.2)  # More generous
            
            # Count interactive patterns
            pattern_matches = 0
            for pattern in interactive_patterns:
                matches = len(re.findall(pattern, text_lower))
                pattern_matches += matches
            
            score += min(0.4, pattern_matches * 0.08)  # More generous
            
            # Direct address indicators (expanded)
            direct_address = ['you', 'your', 'yourself', 'we', 'us', 'our']
            address_count = sum(1 for word in text_lower.split() if word in direct_address)
            score += min(0.1, address_count * 0.01)  # Small bonus
            
            # ✅ Apply self-talk penalty here too
            if self.patterns["self_conversation"].search(text):
                score *= 0.5  # 50% penalty for self-talk in engagement
            
            scores[i] = min(1.0, score)
                
        return scores

    def _compute_clarity_score(self, texts: List[str]) -> torch.Tensor:
        """Compute clarity score (more generous)"""
        batch_size = len(texts)
        scores = torch.zeros(batch_size, device=self.device)
        
        for i, text in enumerate(texts):
            try:
                from textstat import flesch_reading_ease
                flesch_score = flesch_reading_ease(text)
                if flesch_score >= 50:      # More generous threshold
                    scores[i] = 1.0
                elif flesch_score >= 40:    # More generous
                    scores[i] = 0.8
                elif flesch_score >= 30:    # More generous
                    scores[i] = 0.6
                else:
                    scores[i] = 0.4
            except:
                # Fallback: simple sentence length analysis
                word_count = len(text.split())
                sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
                avg_words = word_count / sentence_count
                
                if avg_words <= 20:
                    scores[i] = 0.8
                elif avg_words <= 30:
                    scores[i] = 0.6
                else:
                    scores[i] = 0.4
        
        return scores

    def _compute_completeness_score(self, texts: List[str]) -> torch.Tensor:
        """Compute completeness score (more generous)"""
        batch_size = len(texts)
        scores = torch.zeros(batch_size, device=self.device)
        
        for i, text in enumerate(texts):
            word_count = len(text.split())
            
            # More generous length requirements
            if 50 <= word_count <= 256:    # Expanded range
                base_score = 0.9
            elif 30 <= word_count <= 300:  # Even more generous
                base_score = 0.7
            elif word_count >= 20:         # Minimum threshold
                base_score = 0.2
            else:
                base_score = -0.5
            
            # Bonus for conclusion indicators
            text_lower = text.lower()
            conclusions = ['so', 'therefore', 'in summary', 'overall', 'this explains', 
                          'does this help', 'make sense', 'clear', 'understand', 'click']
            if any(phrase in text_lower for phrase in conclusions):
                base_score += 0.1
            
            scores[i] = min(1.0, base_score)
        
        return scores

    def _compute_jargon_score(self, texts_lower: List[str]) -> torch.Tensor:
        """Compute jargon avoidance score (more generous)"""
        batch_size = len(texts_lower)
        scores = torch.zeros(batch_size, device=self.device)
        
        # Expanded but more reasonable jargon detection
        jargon_terms = {
            'utilize', 'paradigm', 'synergy', 'leverage', 'optimize', 'streamline',
            'methodology', 'framework', 'infrastructure', 'scalable', 'robust',
            'innovative', 'cutting-edge', 'state-of-the-art', 'holistic', 'comprehensive',
            'stakeholder', 'deliverable', 'actionable', 'bandwidth', 'implementation'
        }
        
        for i, text in enumerate(texts_lower):
            words = text.split()
            if not words:
                scores[i] = 1.0
                continue
                
            jargon_count = sum(1 for word in words if word.strip('.,!?;:') in jargon_terms)
            jargon_ratio = jargon_count / len(words)
            
            # More forgiving jargon penalties
            if jargon_ratio == 0:
                scores[i] = 1.0
            elif jargon_ratio <= 0.03:  # Up to 3% is fine
                scores[i] = 0.95
            elif jargon_ratio <= 0.07:  # Up to 7% gets good score
                scores[i] = 0.8
            elif jargon_ratio <= 0.15:  # Up to 15% gets decent score
                scores[i] = 0.6
            else:
                scores[i] = 0.3
        
        return scores

    # Convenience methods for single text scoring
    def score(self, text: str) -> float:
        """Score a single text"""
        return self.batch_score_optimized([text])[0]
    
    def breakdown(self, text: str) -> dict:
        """Get detailed breakdown for a single text"""
        _, breakdowns = self.batch_score_optimized([text], return_breakdown=True)
        breakdown = breakdowns[0]
        
        # ✅ UPDATED: Include negative components in breakdown
        result = {}
        key_mapping = {
            "analogy": "analogy",
            "step": "step", 
            "fundamentals": "fundamentals",
            "engagement": "engagement",
            "clarity": "clarity",
            "completeness": "complete",
            "no_jargon": "nojargon",
            "no_self_talk": "no_self_talk",
            "no_repetition": "no_repetition", 
            "coherence": "coherence",
            "structure": "structure",
            "reward": "reward"
        }
        
        for old_key, new_key in key_mapping.items():
            if old_key in breakdown:
                result[new_key] = breakdown[old_key]
        
        return result


# Initialize the reward function
reward_scorer = FirstPrinciplesRewardV3()
print("✅ Enhanced reward function with negative rewarding initialized")