from langgraph.graph import StateGraph, END
from ..core.state import MoEState
from ..agents.router import RouterAgent
from ..agents.experts.generic import GenericExpert
from ..agents.synthesizer import SynthesizerAgent
from ..llm.providers import LLMFactory
from ..core.config import MoEConfig
from ..utils.cache import ResponseCache


class MoEGraphBuilder:
    """Builder for constructing the MoE LangGraph workflow"""
    
    def __init__(self, config: MoEConfig):
        """Initialize graph builder"""
        self.config = config
        self.agents = {}
        self.cache = None
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
        
        router_llm = LLMFactory.create_provider(
            provider_type,
            api_key,
            self.config.router_config
        )
        
        synthesizer_llm = LLMFactory.create_provider(
            provider_type,
            api_key,
            self.config.synthesizer_config
        )
        
        self.agents['router'] = RouterAgent(
            router_llm,
            list(self.config.expert_configs.keys())
        )
        
        # Create generic experts for each configured expert type
        for expert_name, expert_config in self.config.expert_configs.items():
            expert_llm = LLMFactory.create_provider(
                provider_type,
                api_key,
                expert_config.llm_config
            )
            
            self.agents[expert_name] = GenericExpert(
                expert_type=expert_name,
                llm_provider=expert_llm,
                confidence_threshold=expert_config.confidence_threshold,
                cache=self.cache  # Pass cache to expert
            )
        
        self.agents['synthesizer'] = SynthesizerAgent(synthesizer_llm)
    
    def _route_to_experts(self, state: MoEState) -> list[str]:
        """Determine which expert nodes to route to based on router selection"""
        selected_experts = state.get('selected_experts', [])
        if not selected_experts:
            return ['general']
        return selected_experts
    
    def build(self):
        """Build and compile the LangGraph workflow with conditional routing"""
        workflow = StateGraph(MoEState)
        
        # Add all nodes
        workflow.add_node("router", self.agents['router'].execute)
        
        for expert_name in self.config.expert_configs.keys():
            workflow.add_node(expert_name, self.agents[expert_name].execute)
        
        workflow.add_node("synthesizer", self.agents['synthesizer'].execute)
        
        # Set entry point
        workflow.set_entry_point("router")
        
        # Add conditional routing from router to selected experts
        workflow.add_conditional_edges(
            "router",
            self._route_to_experts,
            {
                expert_name: expert_name
                for expert_name in self.config.expert_configs.keys()
            }
        )
        
        # Connect all experts to synthesizer
        for expert_name in self.config.expert_configs.keys():
            workflow.add_edge(expert_name, "synthesizer")
        
        # Connect synthesizer to END
        workflow.add_edge("synthesizer", END)
        
        return workflow.compile()