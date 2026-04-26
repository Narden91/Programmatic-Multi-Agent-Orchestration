from dataclasses import dataclass, field
from typing import Dict
import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

DEFAULT_LLM_MODEL = "llama-3.1-8b-instant"
DEPRECATED_MODEL_REPLACEMENTS: Dict[str, str] = {
    "llama-3.1-70b-versatile": DEFAULT_LLM_MODEL,
}
GROQ_CHAT_MODELS = [
    DEFAULT_LLM_MODEL,
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]
OPENAI_CHAT_MODELS = ["gpt-4o", "gpt-4o-mini"]
ANTHROPIC_CHAT_MODELS = ["claude-3-5-sonnet-20240620", "claude-3-5-haiku-20241022"]


# ---------------------------------------------------------------------------
# Secret wrapper — prevents API keys from leaking in repr / logs / tracebacks
# ---------------------------------------------------------------------------

class SecretStr:
    """Opaque string wrapper that masks its value in repr/str.

    Use ``.get_secret_value()`` to retrieve the raw string.  This prevents
    accidental exposure in logs, tracebacks, and ``__repr__`` output.
    """

    __slots__ = ("_value",)

    def __init__(self, value: str = "") -> None:
        object.__setattr__(self, "_value", value)

    def get_secret_value(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return "SecretStr('**********')" if self._value else "SecretStr('')"

    def __str__(self) -> str:
        return "**********" if self._value else ""

    def __bool__(self) -> bool:
        return bool(self._value)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SecretStr):
            return self._value == other._value
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)


@dataclass
class LLMConfig:
    """Configuration for individual LLM"""
    model_name: str
    temperature: float = 0.5
    max_tokens: int = 2000
    top_p: float = 1.0
    
    @classmethod
    def from_env(cls, prefix: str):
        """Create LLMConfig from environment variables"""
        default_model = DEFAULT_LLM_MODEL
        return cls(
            model_name=os.getenv(f"{prefix}_MODEL", default_model),
            temperature=float(os.getenv(f"{prefix}_TEMPERATURE", "0.5")),
            max_tokens=int(os.getenv("MAX_TOKENS", "2000"))
        )


@dataclass
class ExpertConfig:
    """Configuration for an expert agent"""
    name: str
    description: str
    llm_config: LLMConfig
    system_prompt: str
    confidence_threshold: float = 0.7
    provider_type: str = ""  # empty → inherit from global config


@dataclass
class MoEConfig:
    """Main configuration for the MoE system"""
    
    groq_api_key: SecretStr = field(default_factory=lambda: SecretStr(os.getenv("GROQ_API_KEY", "")))
    openai_api_key: SecretStr = field(default_factory=lambda: SecretStr(os.getenv("OPENAI_API_KEY", "")))
    anthropic_api_key: SecretStr = field(default_factory=lambda: SecretStr(os.getenv("ANTHROPIC_API_KEY", "")))
    
    orchestrator_config: LLMConfig = field(default_factory=lambda: LLMConfig.from_env("ORCHESTRATOR"))
    expert_configs: Dict[str, ExpertConfig] = field(default_factory=dict)

    # Kept as aliases for backward-compat; both resolve to orchestrator_config.
    @property
    def router_config(self) -> LLMConfig:
        return self.orchestrator_config

    @property
    def synthesizer_config(self) -> LLMConfig:
        return self.orchestrator_config
    
    max_parallel_experts: int = field(default_factory=lambda: int(os.getenv("MAX_PARALLEL_EXPERTS", "4")))
    enable_logging: bool = field(default_factory=lambda: os.getenv("ENABLE_LOGGING", os.getenv("ENABLE_METRICS", "true")).lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    
    request_timeout: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "120")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    retry_delay: int = field(default_factory=lambda: int(os.getenv("RETRY_DELAY", "1")))
    orchestrator_candidate_count: int = field(default_factory=lambda: int(os.getenv("ORCHESTRATOR_CANDIDATES", "1")))
    orchestrator_script_few_shot_count: int = field(default_factory=lambda: int(os.getenv("ORCHESTRATOR_SCRIPT_FEW_SHOTS", "2")))
    orchestrator_atom_few_shot_count: int = field(default_factory=lambda: int(os.getenv("ORCHESTRATOR_ATOM_FEW_SHOTS", "4")))
    enable_atom_few_shot_retrieval: bool = field(default_factory=lambda: os.getenv("ENABLE_ATOM_FEW_SHOT_RETRIEVAL", "true").lower() == "true")
    enable_metadata_selection_bias: bool = field(default_factory=lambda: os.getenv("ENABLE_METADATA_SELECTION_BIAS", "true").lower() == "true")
    registry_db_path: str = field(default_factory=lambda: os.getenv("REGISTRY_DB_PATH", ".moe_registry.db"))

    sandbox_isolate_process: bool = field(
        default_factory=lambda: os.getenv(
            "SANDBOX_ISOLATE_PROCESS",
            # Windows spawn-based multiprocessing is too slow for interactive use;
            # default to in-process execution unless explicitly overridden.
            "false" if os.name == "nt" else "true",
        ).lower() == "true"
    )
    sandbox_max_code_chars: int = field(default_factory=lambda: int(os.getenv("SANDBOX_MAX_CODE_CHARS", "30000")))
    sandbox_max_ast_nodes: int = field(default_factory=lambda: int(os.getenv("SANDBOX_MAX_AST_NODES", "8000")))
    sandbox_max_statements: int = field(default_factory=lambda: int(os.getenv("SANDBOX_MAX_STATEMENTS", "1500")))
    sandbox_max_query_calls: int = field(default_factory=lambda: int(os.getenv("SANDBOX_MAX_QUERY_CALLS", "120")))
    
    enable_cache: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHE", "true").lower() == "true")
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))  # 1 hour default
    cache_max_size: int = field(default_factory=lambda: int(os.getenv("CACHE_MAX_SIZE", "100")))
    
    def __post_init__(self):
        """Initialize default expert configurations"""
        if not self.expert_configs:
            self.expert_configs = self._create_default_expert_configs()
    
    def _create_default_expert_configs(self) -> Dict[str, ExpertConfig]:
        """Create default configurations for all experts"""
        return {
            "technical": ExpertConfig(
                name="technical",
                description="Expert in programming, technology, and sciences",
                llm_config=LLMConfig.from_env("TECHNICAL"),
                system_prompt="You are a technical expert specialized in programming, technology, and sciences.",
                confidence_threshold=0.85
            ),
            "creative": ExpertConfig(
                name="creative",
                description="Expert in storytelling and creative content",
                llm_config=LLMConfig.from_env("CREATIVE"),
                system_prompt="You are a creative expert specialized in storytelling, brainstorming, and original content.",
                confidence_threshold=0.80
            ),
            "analytical": ExpertConfig(
                name="analytical",
                description="Expert in data analysis and logical reasoning",
                llm_config=LLMConfig.from_env("ANALYTICAL"),
                system_prompt="You are an analytical expert specialized in data analysis, logic, and rational decisions.",
                confidence_threshold=0.88
            ),
            "general": ExpertConfig(
                name="general",
                description="Expert in general knowledge and conversation",
                llm_config=LLMConfig.from_env("GENERAL"),
                system_prompt="You are a general knowledge expert, friendly and conversational.",
                confidence_threshold=0.75
            ),
            "critical-thinker": ExpertConfig(
                name="critical-thinker",
                description="Expert in scientific QA, logical fallacy checking, and evidence evaluation",
                llm_config=LLMConfig.from_env("CRITICAL_THINKER"),
                system_prompt="You are a Critical-Thinker expert. Your job is to rigorously evaluate statements, arguments, and evidence for logical fallacies, biases, and structural soundness.",
                confidence_threshold=0.90
            )
        }
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not (self.groq_api_key or self.openai_api_key or self.anthropic_api_key):
            raise ValueError(
                "At least one API key (GROQ_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY) is required."
            )
        
        if not self.expert_configs:
            raise ValueError("At least one expert configuration is required")

        if self.sandbox_max_code_chars <= 0:
            raise ValueError("SANDBOX_MAX_CODE_CHARS must be > 0")
        if self.sandbox_max_ast_nodes <= 0:
            raise ValueError("SANDBOX_MAX_AST_NODES must be > 0")
        if self.sandbox_max_statements <= 0:
            raise ValueError("SANDBOX_MAX_STATEMENTS must be > 0")
        if self.sandbox_max_query_calls <= 0:
            raise ValueError("SANDBOX_MAX_QUERY_CALLS must be > 0")
        if self.orchestrator_candidate_count <= 0:
            raise ValueError("ORCHESTRATOR_CANDIDATES must be > 0")
        if self.orchestrator_candidate_count > 8:
            raise ValueError("ORCHESTRATOR_CANDIDATES must be <= 8 to avoid runaway generation cost")
        if self.orchestrator_script_few_shot_count < 0:
            raise ValueError("ORCHESTRATOR_SCRIPT_FEW_SHOTS must be >= 0")
        if self.orchestrator_atom_few_shot_count < 0:
            raise ValueError("ORCHESTRATOR_ATOM_FEW_SHOTS must be >= 0")
        
        return True
    
    def get_provider_type(self) -> str:
        """Determine which LLM provider to use based on available keys"""
        if self.groq_api_key:
            return "groq"
        elif self.openai_api_key:
            return "openai"
        elif self.anthropic_api_key:
            return "anthropic"
        else:
            raise ValueError("No valid API key found")

    def get_api_key(self, provider_type: str) -> str:
        """Safely retrieve the raw API key string for a given provider type."""
        secret: SecretStr = getattr(self, f"{provider_type}_api_key")
        return secret.get_secret_value()


config = MoEConfig()