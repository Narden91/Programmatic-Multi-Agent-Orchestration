import time
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.core.registry import OrchestrationRegistry

def test_singleton():
    print("--- Instantiating first registry ---")
    start = time.time()
    r1 = OrchestrationRegistry()
    end = time.time()
    print(f"First instantiation took: {end - start:.2f}s")
    
    print("\n--- Instantiating second registry ---")
    start = time.time()
    r2 = OrchestrationRegistry()
    end = time.time()
    print(f"Second instantiation took: {end - start:.4f}s")
    
    if r1.model is r2.model:
        print("\nSUCCESS: Both registries share the same model instance.")
    else:
        print("\nFAILURE: Registries have different model instances.")

if __name__ == "__main__":
    test_singleton()
