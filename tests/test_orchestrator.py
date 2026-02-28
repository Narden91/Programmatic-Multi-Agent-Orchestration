import os
import sys
import asyncio
from pathlib import Path

# Add project root to sys path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import MoEConfig
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder

async def main():
    if not os.getenv("GROQ_API_KEY"):
        print("Set GROQ_API_KEY in environment to run this test.")
        return
        
    config = MoEConfig()
    try:
        config.validate()
    except Exception as e:
        print(f"Config Error: {e}")
        return
        
    builder = MoEGraphBuilder(config)
    graph = builder.build()
    
    state = create_initial_state("Explain the concept of quantum entanglement. Give a creative analogy and a technical explanation.")
    print("Testing programmatic orchestration...")
    
    try:
        result = await graph.ainvoke(state)
        print("\n\n--- Generated Code ---")
        print(result.get('generated_code'))
        
        print(f"\nExecuted in {result.get('code_execution_iterations')} iterations.")
        
        if result.get("code_execution_error"):
            print("\n\n--- Execution Error ---")
            print(result.get("code_execution_error"))
            
        print("\n\n--- Final Answer ---")
        print(result.get('final_answer'))
        
    except Exception as e:
        print(f"\nGraph execution failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
