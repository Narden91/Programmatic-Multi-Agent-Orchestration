from langgraph.graph import StateGraph, END
from ..core.state import MoEState
from ..agents.orchestrator import OrchestratorAgent, CodeExecutionAgent
from ..agents.registry import registry
from ..llm.providers import LLMFactory
from ..core.config import MoEConfig
from ..utils.cache import ResponseCache
from ..utils.script_bank import ScriptBank


class MoEGraphBuilder:
    """Builder for constructing the MoE LangGraph workflow"""
    
    def __init__(self, config: MoEConfig, script_bank: ScriptBank | None = None):
        """Initialize graph builder"""
        self.config = config
        self.agents = {}
        self.cache = None
        self.script_bank = script_bank or ScriptBank()
        self._initialize_cache()
        self._initialize_agents()
    
    def _initialize_cache(self):
        """Initialize response cache if enabled"""
        if self.config.enable_cache:
            self.cache = ResponseCache(
                ttl_seconds=self.config.cache_ttl,
                max_size=self.config.cache_max_size
            )
    
    def _initialize_agents(self):
        """Initialize all agent instances"""
        provider_type = self.config.get_provider_type()
        api_key = getattr(self.config, f"{provider_type}_api_key")
        
        # Reuse orchestrator config for the script-generating LLM
        orchestrator_llm = LLMFactory.create_provider(
            provider_type,
            api_key,
            self.config.orchestrator_config
        )
        
        self.agents['orchestrator'] = OrchestratorAgent(
            orchestrator_llm,
            available_experts=list(registry.types),
            script_bank=self.script_bank,
        )
        
        self.agents['code_executor'] = CodeExecutionAgent(
            timeout_seconds=self.config.request_timeout,
            script_bank=self.script_bank,
        )
    
    def _should_retry_code(self, state: MoEState) -> str:
        """Determine whether to retry if execution failed"""
        error = state.get('code_execution_error')
        iterations = state.get('code_execution_iterations', 0)
        
        max_retries = self.config.max_retries
        if error and iterations < max_retries:
            return "orchestrator"
        return END
    
    def build(self):
        """Build and compile the linear programmatic workflow"""
        workflow = StateGraph(MoEState)
        
        # Add nodes
        workflow.add_node("orchestrator", self.agents['orchestrator'].execute)
        workflow.add_node("code_executor", self.agents['code_executor'].aexecute)
        
        # Set entry point
        workflow.set_entry_point("orchestrator")
        
        # Linear sequence
        workflow.add_edge("orchestrator", "code_executor")
        
        # Add conditional routing for retry mechanism
        workflow.add_conditional_edges(
            "code_executor",
            self._should_retry_code,
            {
                "orchestrator": "orchestrator",
                END: END
            }
        )
        
        return workflow.compile()