import logging
import re
from typing import Dict, Any, Optional, List, Tuple

logger = logging.getLogger(__name__)

# Lightweight MMLU-style multiple-choice questions (subset across domains)
_MMLU_QUESTIONS: List[Tuple[str, List[str], int]] = [
    ("What is the capital of France?", ["London", "Paris", "Berlin", "Madrid"], 1),
    ("Which planet is known as the Red Planet?", ["Venus", "Mars", "Jupiter", "Saturn"], 1),
    ("What is the chemical symbol for gold?", ["Go", "Gd", "Au", "Ag"], 2),
    ("Who wrote 'Romeo and Juliet'?", ["Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"], 1),
    ("What is the derivative of x^2?", ["x", "2x", "x^3/3", "2"], 1),
    ("In which year did World War II end?", ["1943", "1944", "1945", "1946"], 2),
    ("What is the largest ocean on Earth?", ["Atlantic", "Indian", "Arctic", "Pacific"], 3),
    ("Which data structure uses FIFO ordering?", ["Stack", "Queue", "Tree", "Graph"], 1),
    ("What does CPU stand for?", ["Central Processing Unit", "Computer Personal Unit", "Central Program Utility", "Core Processing Unit"], 0),
    ("Which sorting algorithm has O(n log n) average complexity?", ["Bubble Sort", "Insertion Sort", "Merge Sort", "Selection Sort"], 2),
]

# Lightweight GSM8K-style math word problems
_GSM8K_QUESTIONS: List[Tuple[str, float]] = [
    ("A store sells apples at $2 each. If you buy 5 apples, how much do you spend?", 10.0),
    ("A train travels 60 mph for 2.5 hours. How far does it go?", 150.0),
    ("If 3x + 7 = 22, what is x?", 5.0),
    ("A rectangle has length 8 and width 5. What is its area?", 40.0),
    ("A shirt costs $25 after a 20% discount. What was the original price?", 31.25),
    ("How many minutes are in 3.5 hours?", 210.0),
    ("If 15 workers complete a job in 4 hours, how many hours would 5 workers take?", 12.0),
    ("A circle has radius 3. What is its circumference? (use pi=3.14)", 18.84),
]

def _extract_answer(response: str, choices: Optional[List[str]] = None) -> str:
    """Extract a short answer from model response text."""
    response = response.strip()
    if choices:
        for i, choice in enumerate(choices):
            if choice.lower() in response.lower():
                return str(i)
        match = re.search(r'\b([A-D])\b', response)
        if match:
            idx = ord(match.group(1).upper()) - ord('A')
            return str(idx)
    return response

def _extract_number(response: str) -> Optional[float]:
    """Extract a numeric answer from model response text."""
    match = re.search(r'[-+]?\d*\.?\d+', response)
    if match:
        return float(match.group())
    return None

class AblationTaxBenchmarker:
    """
    Measures the 'Ablation Tax' – the cost of model editing/ablation on general intelligence.
    Integrates lightweight standard benchmarks (e.g., MMLU, GSM8K) to enforce the contract:
    an ablation job only succeeds if adversarial robustness exceeds a threshold AND
    general intelligence drops by less than a defined threshold.
    """

    def __init__(self, mmlu_weight: float = 0.5, gsm8k_weight: float = 0.5):
        self.mmlu_weight = mmlu_weight
        self.gsm8k_weight = gsm8k_weight

    def _query_model(self, model: Any, prompt: str, max_tokens: int = 100) -> str:
        """Generate a response from the model (supports PyTorch, GGUF/llama_cpp, and callable interfaces)."""
        if model is None:
            return ""
        if hasattr(model, 'generate'):
            if hasattr(model, 'tokenizer'):
                import torch
                inputs = model.tokenizer(prompt, return_tensors="pt").to(model.device if hasattr(model, 'device') else "cpu")
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
                return model.tokenizer.decode(outputs[0], skip_special_tokens=True)
            else:
                return str(model.generate(prompt, max_tokens=max_tokens))
        elif callable(model):
            result = model(prompt, max_tokens=max_tokens)
            if isinstance(result, dict):
                return result.get('choices', [{}])[0].get('text', '')
            return str(result)
        return ""

    def evaluate_mmlu(self, model: Any) -> float:
        """
        Lightweight MMLU evaluation using a small subset of multiple-choice questions
        across domains (geography, chemistry, math, literature, history, CS).
        Returns accuracy score (0.0 - 1.0).
        """
        logger.info("Evaluating MMLU benchmark (%d questions)...", len(_MMLU_QUESTIONS))
        if model is None:
            logger.warning("MMLU: model is None, returning 0.0")
            return 0.0
        correct = 0
        for question, choices, answer_idx in _MMLU_QUESTIONS:
            prompt = f"{question}\n"
            for i, choice in enumerate(choices):
                prompt += f"  {chr(ord('A')+i)}. {choice}\n"
            prompt += "\nAnswer with the letter of the correct choice."
            response = self._query_model(model, prompt, max_tokens=50)
            extracted = _extract_answer(response, choices)
            if extracted == str(answer_idx):
                correct += 1
        score = correct / len(_MMLU_QUESTIONS)
        logger.info(f"MMLU accuracy: {correct}/{len(_MMLU_QUESTIONS)} = {score:.4f}")
        return score

    def evaluate_gsm8k(self, model: Any) -> float:
        """
        Lightweight GSM8K evaluation using a small subset of grade-school math word problems.
        Returns accuracy score (0.0 - 1.0) based on exact numeric match (within 1% tolerance).
        """
        logger.info("Evaluating GSM8K benchmark (%d problems)...", len(_GSM8K_QUESTIONS))
        if model is None:
            logger.warning("GSM8K: model is None, returning 0.0")
            return 0.0
        correct = 0
        for question, expected_answer in _GSM8K_QUESTIONS:
            prompt = f"{question}\n\nSolve the problem and provide the final numeric answer."
            response = self._query_model(model, prompt, max_tokens=100)
            extracted = _extract_number(response)
            if extracted is not None:
                tolerance = abs(expected_answer) * 0.01
                if abs(extracted - expected_answer) <= tolerance:
                    correct += 1
        score = correct / len(_GSM8K_QUESTIONS)
        logger.info(f"GSM8K accuracy: {correct}/{len(_GSM8K_QUESTIONS)} = {score:.4f}")
        return score

    def compute_general_intelligence(self, model: Any) -> float:
        """
        Computes a composite score of general intelligence based on MMLU and GSM8K.
        """
        mmlu_score = self.evaluate_mmlu(model)
        gsm8k_score = self.evaluate_gsm8k(model)
        
        composite_score = (mmlu_score * self.mmlu_weight) + (gsm8k_score * self.gsm8k_weight)
        logger.info(f"Composite General Intelligence Score: {composite_score:.4f}")
        return composite_score

    def verify_contract(
        self,
        adversarial_robustness: float,
        baseline_model: Any = None,
        edited_model: Any = None,
        robustness_threshold: float = 0.90,
        max_intelligence_drop: float = 0.015,
        baseline_intelligence: Optional[float] = None,
        edited_intelligence: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Enforces the ablation contract:
        1. Adversarial robustness must exceed `robustness_threshold`.
        2. General intelligence drop (baseline - edited) must be less than `max_intelligence_drop`.
        """
        logger.info("Verifying ablation tax contract...")
        
        if baseline_intelligence is None:
            baseline_intelligence = self.compute_general_intelligence(baseline_model)
        
        if edited_intelligence is None:
            edited_intelligence = self.compute_general_intelligence(edited_model)

        intelligence_drop = baseline_intelligence - edited_intelligence
        
        robustness_passed = adversarial_robustness >= robustness_threshold
        tax_passed = intelligence_drop < max_intelligence_drop

        passed = robustness_passed and tax_passed

        result = {
            "status": "ok",
            "passed": passed,
            "metrics": {
                "adversarial_robustness": adversarial_robustness,
                "robustness_threshold": robustness_threshold,
                "baseline_intelligence": baseline_intelligence,
                "edited_intelligence": edited_intelligence,
                "intelligence_drop": intelligence_drop,
                "max_intelligence_drop": max_intelligence_drop
            },
            "contract_evaluation": {
                "robustness_passed": robustness_passed,
                "tax_passed": tax_passed
            }
        }
        
        if not robustness_passed:
            logger.warning(f"Contract Failed: Adversarial robustness ({adversarial_robustness:.4f}) is below threshold ({robustness_threshold:.4f})")
        if not tax_passed:
            logger.warning(f"Contract Failed: General intelligence drop ({intelligence_drop:.4f}) exceeds maximum allowed ({max_intelligence_drop:.4f})")
            
        if passed:
            logger.info("Contract Passed: Ablation job meets all criteria.")

        return result
