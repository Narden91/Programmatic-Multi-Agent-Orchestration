import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import MoEConfig
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder
from src.utils.logging import setup_logger
from src.utils.metrics import PerformanceMetrics
import os
from dotenv import load_dotenv
import time

# Load environment
load_dotenv()

# Setup logging
logger = setup_logger('MoE Demo', 'logs/demo.log')


def print_separator(char='=', length=60):
    """Print a separator line"""
    print(char * length)


def print_section(title: str):
    """Print a section header"""
    print(f"\n{title}")
    print_separator()


def demo_single_expert():
    """Demo: Single expert selection"""
    print_section("DEMO 1: Single Expert Selection")
    
    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY"))
    builder = MoEGraphBuilder(config)
    graph = builder.build()
    
    query = "What is the capital of France?"
    print(f"📝 Query: {query}")
    print(f"🎯 Expected: General expert\n")
    
    state = create_initial_state(query)
    result = graph.invoke(state)
    
    print(f"✅ Selected: {', '.join(result['selected_experts'])}")
    print(f"\n📄 Answer:\n{result['final_answer'][:200]}...")


def demo_multi_expert():
    """Demo: Multiple expert collaboration"""
    print_section("DEMO 2: Multiple Expert Collaboration")
    
    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY"))
    builder = MoEGraphBuilder(config)
    graph = builder.build()
    
    query = "Explain quantum computing with creative analogies"
    print(f"📝 Query: {query}")
    print(f"🎯 Expected: Technical + Creative experts\n")
    
    state = create_initial_state(query)
    result = graph.invoke(state)
    
    print(f"✅ Selected: {', '.join(result['selected_experts'])}")
    print(f"\n🤖 Expert Responses:")
    for expert, response in result['expert_responses'].items():
        print(f"\n  {expert.upper()}:")
        print(f"  {response[:150]}...")
    
    print(f"\n✨ Synthesized Answer:\n{result['final_answer'][:200]}...")


def demo_with_metrics():
    """Demo: Performance metrics tracking"""
    print_section("DEMO 3: Performance Metrics")
    
    metrics = PerformanceMetrics()
    
    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY"))
    builder = MoEGraphBuilder(config)
    graph = builder.build()
    
    test_queries = [
        "Write a Python function for fibonacci",
        "What's the weather like?",
        "Compare sorting algorithms",
    ]
    
    for query in test_queries:
        print(f"\n📝 Processing: {query}")
        start = time.time()
        
        state = create_initial_state(query)
        result = graph.invoke(state)
        
        duration = time.time() - start
        print(f"⏱️  Time: {duration:.2f}s")
        print(f"✅ Selected: {', '.join(result['selected_experts'])}")
        
        metrics.record_execution_time('full_pipeline', duration)
    
    # Print metrics summary
    print("\n📊 Performance Summary:")
    summary = metrics.get_summary()
    for component, stats in summary.items():
        print(f"\n{component}:")
        print(f"  Executions: {stats['executions']}")
        print(f"  Avg Time: {stats['avg_time']:.2f}s")
        print(f"  Min Time: {stats['min_time']:.2f}s")
        print(f"  Max Time: {stats['max_time']:.2f}s")


def demo_reasoning_steps():
    """Demo: Reasoning steps visualization"""
    print_section("DEMO 4: Reasoning Steps")
    
    config = MoEConfig(groq_api_key=os.getenv("GROQ_API_KEY"))
    builder = MoEGraphBuilder(config)
    graph = builder.build()
    
    query = "Explain machine learning to a 5-year-old"
    print(f"📝 Query: {query}\n")
    
    state = create_initial_state(query)
    result = graph.invoke(state)
    
    print("📋 Reasoning Steps:")
    for i, step in enumerate(result['reasoning_steps'], 1):
        print(f"\n{i}. {step['agent']}")
        print(f"   Action: {step['action']}")
        print(f"   Time: {step['timestamp']}")
        if step.get('details'):
            print(f"   Details: {step['details']}")


def main():
    """Run all demos"""
    print_separator('=', 80)
    print("🧠 MIXTURE OF EXPERTS - COMPREHENSIVE DEMO")
    print_separator('=', 80)
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment")
        print("Please set it in your .env file")
        return
    
    try:
        # Run demos
        demo_single_expert()
        time.sleep(2)
        
        demo_multi_expert()
        time.sleep(2)
        
        demo_with_metrics()
        time.sleep(2)
        
        demo_reasoning_steps()
        
        print_section("🎉 Demo Completed!")
        print("\n💡 Key Takeaways:")
        print("  1. Router intelligently selects appropriate experts")
        print("  2. Multiple experts can collaborate on complex queries")
        print("  3. System tracks performance metrics")
        print("  4. Full reasoning transparency with step logging")
        print("  5. Modular architecture allows easy extension")
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()