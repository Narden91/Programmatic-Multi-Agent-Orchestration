from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Dict, Any, List, Optional
import time
from langchain_core.messages import SystemMessage, HumanMessage

from .config import config, ExpertConfig
from ..llm.providers import LLMFactory
from ..utils.metrics import get_token_tracker

@dataclass(frozen=True)
class SemanticAtom:
    """Compact semantic unit produced by an expert response."""

    atom_id: str
    text: str
    confidence: float = 0.5
    dependencies: List[str] = field(default_factory=list)
    evidence_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        digest = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        return digest[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "text": self.text,
            "confidence": self.confidence,
            "dependencies": list(self.dependencies),
            "evidence_tags": list(self.evidence_tags),
            "metadata": dict(self.metadata),
            "content_hash": self.content_hash,
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], *, fallback_prefix: str = "atom") -> "SemanticAtom":
        text = str(
            payload.get("compressed_text")
            or payload.get("text")
            or payload.get("claim")
            or ""
        ).strip()
        atom_id = str(payload.get("claim_id") or payload.get("id") or "").strip()
        if not atom_id:
            atom_id = f"{fallback_prefix}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:8]}"

        dependencies = payload.get("dependencies") or []
        if not isinstance(dependencies, list):
            dependencies = []

        evidence_tags = payload.get("evidence_tags") or payload.get("tags") or []
        if not isinstance(evidence_tags, list):
            evidence_tags = []

        try:
            confidence = float(payload.get("confidence", 0.5))
        except (TypeError, ValueError):
            confidence = 0.5

        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"claim_id", "id", "compressed_text", "text", "claim", "confidence", "dependencies", "evidence_tags", "tags"}
        }

        return cls(
            atom_id=atom_id,
            text=text,
            confidence=max(0.0, min(confidence, 1.0)),
            dependencies=[str(item) for item in dependencies],
            evidence_tags=[str(item) for item in evidence_tags],
            metadata=metadata,
        )


@dataclass
class AgentResult:
    """Standardized result returned by a micro-agent."""
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    token_count: int = 0
    duration_ms: int = 0
    atoms: List[SemanticAtom] = field(default_factory=list)

    def clone(self, *, duration_ms: Optional[int] = None) -> "AgentResult":
        return AgentResult(
            text=self.text,
            metadata=dict(self.metadata),
            token_count=self.token_count,
            duration_ms=self.duration_ms if duration_ms is None else duration_ms,
            atoms=list(self.atoms),
        )

    @classmethod
    def from_response_text(
        cls,
        response_text: str,
        *,
        agent_type: str,
        metadata: Optional[Dict[str, Any]] = None,
        token_count: int = 0,
        duration_ms: int = 0,
    ) -> "AgentResult":
        text, atoms, parse_meta = _parse_agent_response(response_text, agent_type=agent_type)
        merged_metadata = dict(metadata or {})
        merged_metadata.update(parse_meta)
        return cls(
            text=text,
            metadata=merged_metadata,
            token_count=token_count,
            duration_ms=duration_ms,
            atoms=atoms,
        )

_prompt_cache: Dict[str, AgentResult] = {}


def _parse_agent_response(response_text: str, *, agent_type: str) -> tuple[str, List[SemanticAtom], Dict[str, Any]]:
    stripped = response_text.strip()
    if not stripped:
        return "", [], {"response_format": "empty", "atom_count": 0}

    payload = _try_parse_json_payload(stripped)
    if isinstance(payload, dict):
        atoms_payload = payload.get("atoms")
        if isinstance(atoms_payload, list):
            atoms = [
                SemanticAtom.from_payload(item, fallback_prefix=agent_type)
                for item in atoms_payload
                if isinstance(item, dict)
            ]
            summary = str(payload.get("summary") or payload.get("text") or "").strip()
            if not summary:
                summary = "\n".join(atom.text for atom in atoms if atom.text).strip()
            if atoms:
                return summary, atoms, {
                    "response_format": "semantic_atoms",
                    "atom_count": len(atoms),
                }

        if _looks_like_atom_payload(payload):
            atom = SemanticAtom.from_payload(payload, fallback_prefix=agent_type)
            text = atom.text
            return text, [atom], {
                "response_format": "semantic_atom",
                "atom_count": 1,
            }

    atom = SemanticAtom(
        atom_id=f"{agent_type}:{hashlib.sha256(stripped.encode('utf-8')).hexdigest()[:8]}",
        text=stripped,
        metadata={"source": "plain_text_fallback"},
    )
    return stripped, [atom], {
        "response_format": "plain_text",
        "atom_count": 1,
    }


def _try_parse_json_payload(text: str) -> Any:
    candidate = text
    fenced = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def _looks_like_atom_payload(payload: Dict[str, Any]) -> bool:
    return any(
        key in payload
        for key in ("claim_id", "compressed_text", "claim", "text")
    )

async def query_agent(agent_type: str, prompt: str, context_ids: Optional[List[str]] = None) -> AgentResult:
    """
    Spawns a transient micro-agent of the requested type, processes the prompt, and returns the result.
    
    Args:
        agent_type: The type of expert to spawn (e.g., 'technical', 'analytical', 'creative', 'general', 'critical-thinker')
        prompt: The task instruction or query.
        context_ids: Optional list of memory context IDs to retrieve and inject.
    """
    cache_key = f"{agent_type}:{prompt}:{','.join(context_ids) if context_ids else ''}"
    if cache_key in _prompt_cache:
        # Clone to avoid mutable sharing
        cached = _prompt_cache[cache_key]
        return cached.clone(duration_ms=0)

    start_time = time.time()
    
    # 1. Retrieve config
    if agent_type not in config.expert_configs:
        # Fallback to general if unknown, or raise. We'll raise to keep it strict.
        raise ValueError(f"Unknown agent type: {agent_type}. Available: {list(config.expert_configs.keys())}")
        
    expert_config = config.expert_configs[agent_type]
    
    # Check if context_ids are provided (memory integration to be added)
    # If we have context, we would look it up here and prepend it to the prompt.
    # For now we'll just mock the context retrieval part since Memory class isn't fully injected yet.
    # In sandbox.py, memory_search would have been used by the orchestrator script to get facts.
    
    full_prompt = prompt
    if context_ids:
        # This assumes the sandbox has already done the `memory_search`. If context_ids is passed,
        # it might just be the actual texts, or the orchestrator should pass the searched texts directly.
        # But per idea_v2, the orchestrator might pass raw context texts or we fetch them here.
        pass

    # 2. Instantiate LLM
    provider_type = expert_config.provider_type or config.get_provider_type()
    api_key = config.get_api_key(provider_type)
    
    llm_provider = LLMFactory.create_provider(
        provider_type=provider_type,
        api_key=api_key,
        config=expert_config.llm_config,
    )
    
    # 3. Prepare messages
    messages = [
        SystemMessage(content=expert_config.system_prompt),
        HumanMessage(content=full_prompt)
    ]
    
    # 4. Invoke LLM
    # Note: LLMProvider ainvoke expects a string or list of messages depending on the underlying langchain implementation.
    # We should pass a list of messages if supported, or a concatenated string.
    # Langchain models `invoke` and `ainvoke` support lists of messages.
    try:
        prompt_input: Any = messages
        response = await llm_provider.ainvoke(prompt_input)
        # response is an AIMessage in langchain
        response_text = response.content
        
        # We can extract token counts if the model returns them in response.response_metadata
        token_count = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_count = response.usage_metadata.get("total_tokens", 0)
        elif hasattr(response, "response_metadata") and "token_usage" in response.response_metadata:
            token_count = response.response_metadata["token_usage"].get("total_tokens", 0)
            
        tracker = get_token_tracker()
        tracker.record_from_response(f"agent_{agent_type}", expert_config.llm_config.model_name, response)
        
    except Exception as e:
        raise RuntimeError(f"Agent '{agent_type}' execution failed: {str(e)}")
        
    duration_ms = int((time.time() - start_time) * 1000)
    
    agent_result = AgentResult.from_response_text(
        response_text,
        agent_type=agent_type,
        metadata={"agent_type": agent_type, "model": expert_config.llm_config.model_name},
        token_count=token_count,
        duration_ms=duration_ms,
    )
    
    _prompt_cache[cache_key] = agent_result
    
    return agent_result
