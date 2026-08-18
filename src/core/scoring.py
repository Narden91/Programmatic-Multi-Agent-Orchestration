import re
from typing import Dict, Any
from .agents import query_agent

class ScriptScorer:
    """
    Evaluates an orchestration script's final execution based on:
    1. Critical-Thinker LLM-as-a-judge score
    2. Execution metrics (latency, token usage)
    3. Compression & description length efficiency (MOSAIC-MoE)
    4. Error presence
    """
    
    def __init__(
        self,
        weight_quality: float = 0.65,
        weight_efficiency: float = 0.25,
        weight_compression: float = 0.10,
    ):
        self.w_quality = weight_quality
        self.w_efficiency = weight_efficiency
        self.w_compression = weight_compression

    async def score_execution(self, query: str, state: Dict[str, Any]) -> float:
        """
        Calculate a final multi-factor score for the orchestration script's execution.
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
            quality_score = 0.5
            
        # 2. Evaluate latency/efficiency
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

        # 3. Evaluate description length & atomization compression
        total_atoms = sum(
            len((t.get("outputs") or {}).get("atoms") or [])
            for t in trace_dna if t.get("type") == "agent"
        )
        compression_score = min(1.0, 0.5 + (total_atoms * 0.1)) if total_atoms > 0 else 0.5
            
        final_score = (
            (quality_score * self.w_quality)
            + (efficiency_score * self.w_efficiency)
            + (compression_score * self.w_compression)
        )
        return min(max(final_score, 0.0), 1.0)
        
    def _extract_score(self, text: str) -> float:
        """Extract the numeric score from the critical-thinker's output."""
        if not text:
            return 0.5

        # 1. JSON score: {"score": 0.85}
        json_match = re.search(r'["\']score["\']\s*:\s*([0-9]*\.?[0-9]+)', text, re.IGNORECASE)
        if json_match:
            try:
                val = float(json_match.group(1))
                if val > 10.0:
                    val /= 100.0
                elif val > 1.0:
                    val /= 10.0
                return min(max(val, 0.0), 1.0)
            except ValueError:
                pass

        # 2. Pattern "SCORE: X / 10" or "SCORE: X" or "Rating: X"
        match = re.search(r'(?:score|rating)\s*[:=]\s*([0-9]*\.?[0-9]+)(?:\s*/\s*([0-9]+))?', text, re.IGNORECASE)
        if match:
            try:
                val = float(match.group(1))
                denom = float(match.group(2)) if match.group(2) else None
                if denom and denom > 0:
                    return min(max(val / denom, 0.0), 1.0)
                if val > 10.0:
                    val /= 100.0
                elif val > 1.0:
                    val /= 10.0
                return min(max(val, 0.0), 1.0)
            except ValueError:
                pass

        return 0.5
