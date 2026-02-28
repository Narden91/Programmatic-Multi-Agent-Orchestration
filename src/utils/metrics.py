import time
import asyncio
from functools import wraps
from typing import Callable, Any


class PerformanceMetrics:
    """Track performance metrics for the MoE system"""
    
    def __init__(self):
        self.metrics = {}
    
    def record_execution_time(self, agent_name: str, duration: float):
        """Record execution time for an agent"""
        if agent_name not in self.metrics:
            self.metrics[agent_name] = {
                'executions': 0,
                'total_time': 0,
                'min_time': float('inf'),
                'max_time': 0
            }
        
        self.metrics[agent_name]['executions'] += 1
        self.metrics[agent_name]['total_time'] += duration
        self.metrics[agent_name]['min_time'] = min(
            self.metrics[agent_name]['min_time'], 
            duration
        )
        self.metrics[agent_name]['max_time'] = max(
            self.metrics[agent_name]['max_time'], 
            duration
        )
    
    def get_average_time(self, agent_name: str) -> float:
        """Get average execution time for an agent"""
        if agent_name not in self.metrics:
            return 0.0
        
        metrics = self.metrics[agent_name]
        return metrics['total_time'] / metrics['executions']
    
    def get_summary(self) -> dict:
        """Get summary of all metrics"""
        summary = {}
        for agent_name, metrics in self.metrics.items():
            summary[agent_name] = {
                'executions': metrics['executions'],
                'avg_time': metrics['total_time'] / metrics['executions'],
                'min_time': metrics['min_time'],
                'max_time': metrics['max_time']
            }
        return summary
    
    def reset(self):
        """Reset all metrics"""
        self.metrics = {}


def measure_time(metrics: PerformanceMetrics, agent_name: str):
    """
    Decorator to measure execution time of agent methods.
    
    Args:
        metrics: PerformanceMetrics instance
        agent_name: Name of the agent
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_execution_time(agent_name, duration)
                return result
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.record_execution_time(agent_name, duration)
                return result
            return sync_wrapper
    return decorator