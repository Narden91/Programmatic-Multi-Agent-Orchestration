"""
Programmatic Multi-Agent Orchestration — Demo Script

Demonstrates the Code-as-Orchestration paradigm: the Orchestrator LLM
writes an async Python script that calls micro-agent tool functions,
and the sandbox executes it.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

from src.core.config import MoEConfig
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder
from src.utils.logging import setup_logger
from src.utils.metrics import PerformanceMetrics, reset_token_tracker

load_dotenv()

logger = setup_logger("MoE Demo", "logs/demo.log")


def _sep(char: str = "=", length: int = 60) -> None:
    print(char * length)


def _section(title: str) -> None:
    print(f"\n{title}")
    _sep()


def _print_token_usage(result: dict) -> None:
    """Print token-usage summary from a pipeline result."""
    usage = result.get("token_usage", {})
    if usage.get("total_tokens"):
        print(f"  Tokens: {usage['total_tokens']:,} "
              f"(in={usage['total_input_tokens']:,} out={usage['total_output_tokens']:,}) "
              f"≈ ${usage['estimated_cost_usd']:.4f}")


async def demo_single_expert() -> None:
    """Demo 1: Simple single-expert query."""
    _section("DEMO 1: Single Expert — General Knowledge")

    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY", ""))
    graph = MoEGraphBuilder(config).build()

    query = "What is the capital of France?"
    print(f"Query: {query}")
    print("Expected: general expert\n")

    reset_token_tracker()
    state = create_initial_state(query)
    result = await graph.ainvoke(state)

    print(f"Experts called: {', '.join(result['selected_experts'])}")
    print(f"\nGenerated code:\n{result['generated_code']}")
    print(f"\nAnswer:\n{result['final_answer'][:300]}...")
    _print_token_usage(result)


async def demo_multi_expert() -> None:
    """Demo 2: Multi-expert parallel query."""
    _section("DEMO 2: Multi-Expert — Technical + Creative")

    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY", ""))
    graph = MoEGraphBuilder(config).build()

    query = "Explain quantum computing with creative analogies"
    print(f"Query: {query}")
    print("Expected: technical + creative experts\n")

    reset_token_tracker()
    state = create_initial_state(query)
    result = await graph.ainvoke(state)

    print(f"Experts called: {', '.join(result['selected_experts'])}")
    print(f"\nGenerated code:\n{result['generated_code']}")
    for expert, response in result["expert_responses"].items():
        print(f"\n  [{expert.upper()}]: {response[:150]}...")
    print(f"\nFinal answer:\n{result['final_answer'][:300]}...")
    _print_token_usage(result)

    # Show execution plan
    plan = result.get("execution_plan", {})
    if plan.get("calls"):
        print(f"\n  Execution plan: {', '.join(plan['experts_used'])}")
        print(f"  Parallel groups: {plan.get('gather_groups', 0)}")


async def demo_with_metrics() -> None:
    """Demo 3: Performance + token metrics."""
    _section("DEMO 3: Performance & Token Metrics")

    metrics = PerformanceMetrics()
    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY", ""))
    graph = MoEGraphBuilder(config).build()

    queries = [
        "Write a Python function for fibonacci",
        "What is the meaning of life?",
        "Compare sorting algorithms",
    ]

    for query in queries:
        print(f"\nProcessing: {query}")
        reset_token_tracker()
        start = time.time()
        state = create_initial_state(query)
        result = await graph.ainvoke(state)
        duration = time.time() - start
        print(f"  Time: {duration:.2f}s | Experts: {', '.join(result['selected_experts'])}")
        _print_token_usage(result)
        metrics.record_execution_time("full_pipeline", duration)

    print("\nPerformance Summary:")
    for component, stats in metrics.get_summary().items():
        print(f"  {component}: avg={stats['avg_time']:.2f}s  "
              f"min={stats['min_time']:.2f}s  max={stats['max_time']:.2f}s")


async def demo_dynamic_expert() -> None:
    """Demo 4: Register a custom expert at runtime."""
    _section("DEMO 4: Dynamic Expert Registration")

    from src.agents.tools import register_expert

    register_expert(
        expert_type="philosophical",
        description="philosophy, ethics, existential questions",
        prompt_template=(
            "You are a philosopher with deep knowledge of ethics and "
            "existentialism.\n\n"
            'Query: "{query}"\n\n'
            "Reflect deeply and provide a thoughtful philosophical response:\n\n"
            "Response:"
        ),
    )
    print("Registered new 'philosophical' expert")

    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY", ""))
    graph = MoEGraphBuilder(config).build()

    query = "Is free will real? Discuss from a philosophical and technical perspective."
    print(f"Query: {query}\n")

    reset_token_tracker()
    state = create_initial_state(query)
    result = await graph.ainvoke(state)

    print(f"Experts called: {', '.join(result['selected_experts'])}")
    print(f"\nAnswer:\n{result['final_answer'][:400]}...")
    _print_token_usage(result)


async def main() -> None:
    _sep("=", 80)
    print("PROGRAMMATIC MULTI-AGENT ORCHESTRATION — DEMO")
    _sep("=", 80)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found. Set it in .env")
        return

    try:
        await demo_single_expert()
        await asyncio.sleep(1)
        await demo_multi_expert()
        await asyncio.sleep(1)
        await demo_with_metrics()
        await asyncio.sleep(1)
        await demo_dynamic_expert()

        _section("Demo Completed!")
        print("Key Takeaways:")
        print("  1. Orchestrator dynamically writes async Python scripts")
        print("  2. Sandbox executes code with hardened security")
        print("  3. Experts are spawned as tool functions, not graph nodes")
        print("  4. Parallel execution via asyncio.gather()")
        print("  5. Token-level cost accounting per request")
        print("  6. Custom experts can be registered at runtime")
        print("  7. Script bank stores successes for few-shot prompting")
        print("  8. AST analysis extracts execution plans from code")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())