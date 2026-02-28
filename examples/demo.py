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
from src.utils.metrics import PerformanceMetrics

load_dotenv()

logger = setup_logger("MoE Demo", "logs/demo.log")


def _sep(char: str = "=", length: int = 60) -> None:
    print(char * length)


def _section(title: str) -> None:
    print(f"\n{title}")
    _sep()


async def demo_single_expert() -> None:
    """Demo 1: Simple single-expert query."""
    _section("DEMO 1: Single Expert — General Knowledge")

    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY", ""))
    graph = MoEGraphBuilder(config).build()

    query = "What is the capital of France?"
    print(f"Query: {query}")
    print("Expected: general expert\n")

    state = create_initial_state(query)
    result = await graph.ainvoke(state)

    print(f"Experts called: {', '.join(result['selected_experts'])}")
    print(f"\nGenerated code:\n{result['generated_code']}")
    print(f"\nAnswer:\n{result['final_answer'][:300]}...")


async def demo_multi_expert() -> None:
    """Demo 2: Multi-expert parallel query."""
    _section("DEMO 2: Multi-Expert — Technical + Creative")

    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY", ""))
    graph = MoEGraphBuilder(config).build()

    query = "Explain quantum computing with creative analogies"
    print(f"Query: {query}")
    print("Expected: technical + creative experts\n")

    state = create_initial_state(query)
    result = await graph.ainvoke(state)

    print(f"Experts called: {', '.join(result['selected_experts'])}")
    print(f"\nGenerated code:\n{result['generated_code']}")
    for expert, response in result["expert_responses"].items():
        print(f"\n  [{expert.upper()}]: {response[:150]}...")
    print(f"\nFinal answer:\n{result['final_answer'][:300]}...")


async def demo_with_metrics() -> None:
    """Demo 3: Performance metrics."""
    _section("DEMO 3: Performance Metrics")

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
        start = time.time()
        state = create_initial_state(query)
        result = await graph.ainvoke(state)
        duration = time.time() - start
        print(f"  Time: {duration:.2f}s | Experts: {', '.join(result['selected_experts'])}")
        metrics.record_execution_time("full_pipeline", duration)

    print("\nPerformance Summary:")
    for component, stats in metrics.get_summary().items():
        print(f"  {component}: avg={stats['avg_time']:.2f}s  "
              f"min={stats['min_time']:.2f}s  max={stats['max_time']:.2f}s")


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

        _section("Demo Completed!")
        print("Key Takeaways:")
        print("  1. Orchestrator dynamically writes async Python scripts")
        print("  2. Sandbox executes code with hardened security")
        print("  3. Experts are spawned as tool functions, not graph nodes")
        print("  4. Parallel execution via asyncio.gather()")
        print("  5. Full reasoning transparency + code visibility")

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\nError: {e}")


if __name__ == "__main__":
    asyncio.run(main())