from langgraph.graph import END, StateGraph

from ..agents.orchestrator import CodeExecutionAgent, OrchestratorAgent
from ..agents.registry import registry
from ..core.config import MoEConfig
from ..core.sandbox import SandboxPolicy
from ..core.state import MoEState
from ..llm.providers import LLMFactory


class MoEGraphBuilder:
    """Builder for constructing the MoE LangGraph workflow"""

    def __init__(
        self,
        config: MoEConfig,
    ):
        """Initialize graph builder"""
        self.config = config
        self.agents = {}
        self._initialize_agents()

    def _initialize_agents(self):
        provider_type = self.config.get_provider_type()
        api_key = self.config.get_api_key(provider_type)

        orchestrator_llm = LLMFactory.create_provider(
            provider_type,
            api_key,
            self.config.orchestrator_config,
        )

        self.agents["orchestrator"] = OrchestratorAgent(
            orchestrator_llm,
            available_experts=list(registry.types),
            candidate_count=self.config.orchestrator_candidate_count,
            script_few_shot_count=self.config.orchestrator_script_few_shot_count,
            atom_few_shot_count=self.config.orchestrator_atom_few_shot_count,
            enable_atom_few_shot_retrieval=self.config.enable_atom_few_shot_retrieval,
            enable_metadata_selection_bias=self.config.enable_metadata_selection_bias,
            registry_db_path=self.config.registry_db_path,
        )

        self.agents["code_executor"] = CodeExecutionAgent(
            timeout_seconds=self.config.request_timeout,
            isolate_process=self.config.sandbox_isolate_process,
            sandbox_policy=SandboxPolicy(
                max_code_chars=self.config.sandbox_max_code_chars,
                max_ast_nodes=self.config.sandbox_max_ast_nodes,
                max_statements=self.config.sandbox_max_statements,
                max_query_calls=self.config.sandbox_max_query_calls,
            ),
            registry_db_path=self.config.registry_db_path,
        )

    def _should_retry_code(self, state: MoEState) -> str:
        error = state.get("code_execution_error")
        iterations = state.get("code_execution_iterations", 0)

        max_retries = self.config.max_retries
        if error and iterations < max_retries:
            return "orchestrator"
        return END

    def build(self):
        workflow = StateGraph(MoEState)

        workflow.add_node("orchestrator", self.agents["orchestrator"].execute)
        workflow.add_node("code_executor", self.agents["code_executor"].aexecute)
        workflow.set_entry_point("orchestrator")
        workflow.add_edge("orchestrator", "code_executor")

        workflow.add_conditional_edges(
            "code_executor",
            self._should_retry_code,
            {
                "orchestrator": "orchestrator",
                END: END,
            },
        )

        return workflow.compile()
