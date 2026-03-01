"""
Dynamic expert registry — register / unregister expert types at runtime.

Every entry holds enough information for the system to:
  * create prompts for the expert
  * generate sandbox tool functions
  * build orchestrator prompt descriptions

Usage::

    from src.agents.registry import registry

    registry.register(
        expert_type="legal",
        description="contract law, compliance, regulation",
        system_prompt="You are a legal expert…",
        prompt_template='You are a legal expert.\\n\\nQuery: "{query}"\\n\\nRespond:',
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ExpertSpec:
    """Everything the system needs to know about an expert type."""

    expert_type: str
    description: str
    system_prompt: str
    prompt_template: str  # must contain ``{query}`` placeholder
    confidence_threshold: float = 0.75


class ExpertRegistry:
    """Singleton-style registry of expert specifications."""

    def __init__(self) -> None:
        self._specs: Dict[str, ExpertSpec] = {}

    # ---- Mutation -------------------------------------------------------

    def register(
        self,
        expert_type: str,
        description: str,
        system_prompt: str,
        prompt_template: str,
        confidence_threshold: float = 0.75,
    ) -> None:
        """Register (or overwrite) an expert specification."""
        self._specs[expert_type] = ExpertSpec(
            expert_type=expert_type,
            description=description,
            system_prompt=system_prompt,
            prompt_template=prompt_template,
            confidence_threshold=confidence_threshold,
        )

    def unregister(self, expert_type: str) -> None:
        """Remove an expert type from the registry."""
        self._specs.pop(expert_type, None)

    # ---- Queries --------------------------------------------------------

    def get(self, expert_type: str) -> Optional[ExpertSpec]:
        return self._specs.get(expert_type)

    @property
    def types(self) -> List[str]:
        return list(self._specs.keys())

    @property
    def specs(self) -> Dict[str, ExpertSpec]:
        return dict(self._specs)

    def descriptions(self) -> Dict[str, str]:
        """Return ``{expert_type: description}`` for prompt generation."""
        return {s.expert_type: s.description for s in self._specs.values()}

    # ---- Prompt helper --------------------------------------------------

    def create_prompt(self, expert_type: str, query: str) -> str:
        """Format the prompt template for *expert_type* with the given *query*."""
        spec = self._specs.get(expert_type)
        if spec is None:
            raise ValueError(
                f"Unknown expert type: {expert_type}. "
                f"Registered: {list(self._specs.keys())}"
            )
        return spec.prompt_template.format(query=query)

    def __contains__(self, expert_type: str) -> bool:
        return expert_type in self._specs

    def __len__(self) -> int:
        return len(self._specs)


# ---- Module-level singleton ------------------------------------------------

registry = ExpertRegistry()


def _register_defaults() -> None:
    """Populate the registry with the four built-in expert types."""
    _defaults = [
        ExpertSpec(
            expert_type="technical",
            description="programming, technology, mathematics, sciences",
            system_prompt=(
                "You are a technical expert specialized in programming, "
                "technology, and sciences."
            ),
            prompt_template=(
                "You are a technical expert specialized in programming, "
                "technology, and sciences.\n\n"
                'Query: "{query}"\n\n'
                "Provide a response that is:\n"
                "- Precise and detailed\n"
                "- Includes concrete examples if relevant\n"
                "- Uses appropriate technical terminology\n"
                "- Follows best practices\n\n"
                "Response:"
            ),
            confidence_threshold=0.85,
        ),
        ExpertSpec(
            expert_type="creative",
            description="brainstorming, storytelling, creative content",
            system_prompt=(
                "You are a creative expert specialized in storytelling, "
                "brainstorming, and original content."
            ),
            prompt_template=(
                "You are a creative expert specialized in storytelling, "
                "brainstorming, and original content.\n\n"
                'Query: "{query}"\n\n'
                "Provide a response that is:\n"
                "- Innovative and original\n"
                "- Engaging and interesting\n"
                "- Uses creative metaphors or analogies\n"
                "- Thinks outside the box\n\n"
                "Response:"
            ),
            confidence_threshold=0.80,
        ),
        ExpertSpec(
            expert_type="analytical",
            description="data analysis, comparisons, logical decisions",
            system_prompt=(
                "You are an analytical expert specialized in data analysis, "
                "logic, and rational decisions."
            ),
            prompt_template=(
                "You are an analytical expert specialized in data analysis, "
                "logic, and rational decisions.\n\n"
                'Query: "{query}"\n\n'
                "Provide a response that is:\n"
                "- Structured and methodical\n"
                "- Includes pros/cons if applicable\n"
                "- Based on data and facts\n"
                "- Uses step-by-step reasoning\n\n"
                "Response:"
            ),
            confidence_threshold=0.88,
        ),
        ExpertSpec(
            expert_type="general",
            description="general conversation, facts, basic information",
            system_prompt=(
                "You are a general knowledge expert, friendly and conversational."
            ),
            prompt_template=(
                "You are a general knowledge expert, friendly and "
                "conversational.\n\n"
                'Query: "{query}"\n\n'
                "Provide a response that is:\n"
                "- Clear and understandable\n"
                "- Friendly and conversational\n"
                "- Complete but concise\n"
                "- Suitable for all audiences\n\n"
                "Response:"
            ),
            confidence_threshold=0.75,
        ),
        ExpertSpec(
            expert_type="critical-thinker",
            description="scientific QA, logical fallacy checking, evidence evaluation",
            system_prompt=(
                "You are a Critical-Thinker expert. Your job is to rigorously evaluate statements, "
                "arguments, and evidence for logical fallacies, biases, and structural soundness."
            ),
            prompt_template=(
                "You are a Critical-Thinker expert evaluating the following:\n\n"
                '"{query}"\n\n'
                "Analyze it strictly for:\n"
                "- Logical fallacies\n"
                "- Quality of evidence\n"
                "- Potential biases or blind spots\n"
                "- Provide a final quantitative Quality Score (0.0 to 1.0) on the last line like 'SCORE: 0.95'.\n\n"
                "Response:"
            ),
            confidence_threshold=0.90,
        ),
    ]
    for spec in _defaults:
        registry.register(
            expert_type=spec.expert_type,
            description=spec.description,
            system_prompt=spec.system_prompt,
            prompt_template=spec.prompt_template,
            confidence_threshold=spec.confidence_threshold,
        )


_register_defaults()
