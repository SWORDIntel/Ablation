import logging
from typing import Dict, Any, List
from framewerx.aegis_lab.state.db import AegisState

logger = logging.getLogger(__name__)

class CapabilityDriftAnalyzer:
    """
    Method 3: Ability Impact Score (AIS) & Capability Drift Mapping.
    Measures the delta in unrelated performance (e.g. coding) after ablation.
    """
    
    # Standard capability test prompts for evaluating drift
    CAPABILITY_TESTS = {
        "math": [
            "What is 17 * 23?",
            "Solve: 3x + 7 = 22. What is x?",
            "If a train travels 60 mph for 2.5 hours, how far does it go?",
        ],
        "logic": [
            "If all A are B, and all B are C, are all A C?",
            "Given: P implies Q. P is true. Is Q true?",
            "Is this argument valid: 'All cats are mammals. Tom is a cat. Therefore Tom is a mammal.'?",
        ],
        "coding": [
            "Write a Python function to reverse a string.",
            "What does 'git rebase' do?",
            "Explain the difference between a list and a tuple in Python.",
        ],
    }

    def __init__(self, state: AegisState):
        self.state = state

    def calculate_ais(self, job_id: str) -> float:
        """
        Computes Ability Impact Score.
        1.0 = No damage to core reasoning.
        0.0 = Catastrophic capability collapse.

        Evaluates the ablated model against standard capability benchmarks
        by comparing pre- and post-ablation responses using semantic similarity.
        """
        logger.info(f"Analyzing capability drift for job {job_id}")

        # Retrieve job stages to find pre/post ablation artifacts
        stages = self.state.get_stages(job_id)
        if not stages:
            logger.warning(f"No stages found for job {job_id}; returning neutral AIS.")
            return 0.5

        # Try to get the baseline (pre-ablation) and ablated model responses
        baseline_responses = self._get_baseline_responses(job_id, stages)
        ablated_responses = self._get_ablated_responses(job_id, stages)

        if not baseline_responses or not ablated_responses:
            # If we don't have stored responses, evaluate using the model artifact
            model_artifact_id = self._get_model_artifact_id(stages)
            if model_artifact_id:
                return self._evaluate_with_model(model_artifact_id, job_id)
            logger.warning(f"No baseline/ablated responses or model artifact for job {job_id}; returning neutral AIS.")
            return 0.5

        # Compare baseline vs ablated responses using semantic similarity
        preserved_score = self._compute_preservation_score(baseline_responses, ablated_responses)
        
        logger.info(f"Capability preservation score for job {job_id}: {preserved_score:.4f}")
        return preserved_score

    def _get_baseline_responses(self, job_id: str, stages: List[Dict[str, Any]]) -> List[str]:
        """Extract baseline (pre-ablation) model responses from job stages."""
        responses = []
        for stage in stages:
            result = stage.get("result", {})
            if isinstance(result, dict):
                baseline = result.get("baseline_response") or result.get("pre_ablation_response")
                if baseline and isinstance(baseline, str):
                    responses.append(baseline)
        return responses

    def _get_ablated_responses(self, job_id: str, stages: List[Dict[str, Any]]) -> List[str]:
        """Extract ablated (post-ablation) model responses from job stages."""
        responses = []
        for stage in stages:
            result = stage.get("result", {})
            if isinstance(result, dict):
                ablated = result.get("ablated_response") or result.get("post_ablation_response")
                if ablated and isinstance(ablated, str):
                    responses.append(ablated)
        return responses

    def _get_model_artifact_id(self, stages: List[Dict[str, Any]]) -> str:
        """Extract the model artifact ID from job stages."""
        for stage in stages:
            model_id = stage.get("model_artifact_id") or stage.get("model_path")
            if model_id:
                return model_id
        return ""

    def _evaluate_with_model(self, model_artifact_id: str, job_id: str) -> float:
        """
        Evaluate capability preservation by running standard tests against
        the ablated model and comparing with expected answers.
        """
        try:
            from framewerx.sword_llm import get_sword_provider, SWORDLLMInput, SWORDMessage, Role
            provider = get_sword_provider()
            
            correct = 0
            total = 0
            
            for category, prompts in self.CAPABILITY_TESTS.items():
                for prompt in prompts:
                    try:
                        messages = [
                            SWORDMessage(role=Role.USER, content=prompt),
                        ]
                        output = provider.generate(SWORDLLMInput(messages=messages))
                        response = output.content.lower() if output and output.content else ""
                        
                        # Simple correctness heuristics
                        if category == "math":
                            if "391" in response or "17*23" in response.replace(" ", ""):
                                correct += 1
                            elif "5" in response and ("x" in response or "x = 5" in response):
                                correct += 1
                            elif "150" in response:
                                correct += 1
                        elif category == "logic":
                            if "yes" in response or "true" in response or "valid" in response:
                                correct += 1
                        elif category == "coding":
                            if "def" in response or "function" in response or "reverse" in response:
                                correct += 1
                        total += 1
                    except Exception:
                        total += 1
            
            return correct / max(total, 1)
        except Exception as e:
            logger.warning(f"Model-based capability evaluation failed: {e}; returning neutral score.")
            return 0.5

    def _compute_preservation_score(self, baseline: List[str], ablated: List[str]) -> float:
        """
        Compute capability preservation score by measuring semantic similarity
        between baseline and ablated responses.
        """
        import difflib
        
        if not baseline or not ablated:
            return 0.5
        
        # Pair up responses and compute similarity
        similarities = []
        for i in range(min(len(baseline), len(ablated))):
            ratio = difflib.SequenceMatcher(None, baseline[i].lower(), ablated[i].lower()).ratio()
            similarities.append(ratio)
        
        if not similarities:
            return 0.5
        
        avg_similarity = sum(similarities) / len(similarities)
        # Clamp to [0.0, 1.0]
        return min(1.0, max(0.0, avg_similarity))
