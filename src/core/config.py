from dataclasses import dataclass, field
from typing import Dict
import os
from pathlib import Path
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=False)


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
        default_model = "llama-3.3-70b-versatile"
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
    
    request_timeout: int = field(default_factory=lambda: int(os.getenv("REQUEST_TIMEOUT", "60")))
    max_retries: int = field(default_factory=lambda: int(os.getenv("MAX_RETRIES", "3")))
    retry_delay: int = field(default_factory=lambda: int(os.getenv("RETRY_DELAY", "1")))
    
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
            )
        }
    
    def validate(self) -> bool:
        """Validate configuration"""
        if not self.groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is required. "
                "Get your free key at https://console.groq.com/keys"
            )
        
        if not self.expert_configs:
            raise ValueError("At least one expert configuration is required")
        
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