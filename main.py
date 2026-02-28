"""CLI entry-point for the Programmatic Multi-Agent Orchestration system."""

import asyncio
import argparse
import os
import sys
from dotenv import load_dotenv

load_dotenv()


async def run_query(query: str, model: str | None = None) -> None:
    """Run a single query through the programmatic orchestration pipeline."""
    from src.core.config import MoEConfig
    from src.core.state import create_initial_state
    from src.graph.builder import MoEGraphBuilder

    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY", ""))
    if model:
        config.orchestrator_config.model_name = model
        for ec in config.expert_configs.values():
            ec.llm_config.model_name = model
    config.validate()

    graph = MoEGraphBuilder(config).build()
    state = create_initial_state(query)
    result = await graph.ainvoke(state)

    print("\n--- Generated Orchestration Code ---")
    print(result.get("generated_code", "(none)"))
    print("\n--- Experts Called ---")
    print(", ".join(result.get("selected_experts", [])))
    print("\n--- Final Answer ---")
    print(result.get("final_answer", "(no answer)"))

    iterations = result.get("code_execution_iterations", 0)
    if iterations > 1:
        print(f"\n(sandbox retried {iterations} time(s))")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Programmatic Multi-Agent Orchestration CLI",
    )
    parser.add_argument("query", nargs="?", help="The query to process")
    parser.add_argument(
        "--model", "-m", default=None,
        help="Override the LLM model name for all agents",
    )
    parser.add_argument(
        "--ui", action="store_true",
        help="Launch the Streamlit UI instead of the CLI",
    )
    args = parser.parse_args()

    if args.ui:
        os.system(f"{sys.executable} -m streamlit run ui/app.py")
        return

    if not args.query:
        parser.print_help()
        sys.exit(1)

    asyncio.run(run_query(args.query, model=args.model))


if __name__ == "__main__":
    main()
