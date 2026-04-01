import re
from typing import Dict, Any
from .agents import query_agent

class ScriptScorer:
    """
    Evaluates an orchestration script's final execution based on:
    1. Critical-Thinker LLM-as-a-judge score
    2. Execution metrics (latency, token usage)
    3. Error presence
    """
    
    def __init__(self, weight_quality: float = 0.7, weight_efficiency: float = 0.3):
        self.w_quality = weight_quality
        self.w_efficiency = weight_efficiency

    async def score_execution(self, query: str, state: Dict[str, Any]) -> float:
        """
        Calculate a final score for the orchestration script's execution.
        """
        if state.get("code_execution_error"):
            return 0.0  # Failed scripts get 0

        final_answer = state.get("final_answer", "")
        if not final_answer:
            return 0.1  # Success but no generic answer

        # 1. Evaluate quality
        evaluation_prompt = f"Original Query: {query}\n\nFinal Output: {final_answer}"
        try:
            eval_result = await query_agent("critical-thinker", evaluation_prompt)
            quality_score = self._extract_score(eval_result.text)
        except Exception:
            # Fallback score if critical-thinker fails
            quality_score = 0.5
            
        # 2. Evaluate efficiency
        # We prefer scripts with fewer tokens and lower latency, 
        # but the baseline is somewhat arbitrary.
        # Let's just create a normalized heuristic:
        # e.g., if total duration < 5000ms, it's very efficient (score=1.0)
        trace_dna = state.get("trace_dna", [])
        total_duration = sum(t.get("durationMs", 0) for t in trace_dna if t.get("type") == "agent")
        
        if total_duration <= 0:
            efficiency_score = 0.5
        elif total_duration < 2000:
            efficiency_score = 1.0
        elif total_duration < 8000:
            efficiency_score = 0.8
        elif total_duration < 20000:
            efficiency_score = 0.5
        else:
            efficiency_score = 0.2
            
        final_score = (quality_score * self.w_quality) + (efficiency_score * self.w_efficiency)
        return min(max(final_score, 0.0), 1.0)
        
    def _extract_score(self, text: str) -> float:
        """Extract the numeric score from the critical-thinker's output."""
        match = re.search(r'SCORE:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
        if match:
            try:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)
            except ValueError:
                pass
        return 0.5 # Default middle ground
