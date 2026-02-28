from dataclasses import dataclass, field
from typing import Dict
import os
from dotenv import load_dotenv

load_dotenv()


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
    
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    
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


config = MoEConfig()