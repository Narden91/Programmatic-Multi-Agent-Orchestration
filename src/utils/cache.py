"""
Response caching utility for MoE system
"""
import hashlib
import json
import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta


class ResponseCache:
    """LRU cache with TTL for LLM responses"""
    
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 100):
        """
        Initialize response cache.
        
        Args:
            ttl_seconds: Time to live for cached responses in seconds (default: 1 hour)
            max_size: Maximum number of cached items
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.access_times: Dict[str, float] = {}
    
    def _generate_key(self, query: str, expert_type: str) -> str:
        """Generate cache key from query and expert type"""
        content = f"{expert_type}:{query}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, query: str, expert_type: str) -> Optional[str]:
        """
        Get cached response if available and not expired.
        
        Args:
            query: The user query
            expert_type: Type of expert (technical, creative, etc.)
            
        Returns:
            Cached response or None if not found/expired
        """
        key = self._generate_key(query, expert_type)
        
        if key not in self.cache:
            return None
        
        cached_item = self.cache[key]
        cache_time = cached_item['timestamp']
        
        # Check if expired
        if time.time() - cache_time > self.ttl_seconds:
            del self.cache[key]
            del self.access_times[key]
            return None
        
        # Update access time for LRU
        self.access_times[key] = time.time()
        
        return cached_item['response']
    
    def set(self, query: str, expert_type: str, response: str) -> None:
        """
        Cache a response.
        
        Args:
            query: The user query
            expert_type: Type of expert
            response: The LLM response to cache
        """
        key = self._generate_key(query, expert_type)
        
        # Evict least recently used if at max size
        if len(self.cache) >= self.max_size and key not in self.cache:
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            del self.cache[lru_key]
            del self.access_times[lru_key]
        
        self.cache[key] = {
            'response': response,
            'timestamp': time.time(),
            'query': query,
            'expert_type': expert_type
        }
        self.access_times[key] = time.time()
    
    def clear(self) -> None:
        """Clear all cached responses"""
        self.cache.clear()
        self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'ttl_seconds': self.ttl_seconds,
            'oldest_entry': min([v['timestamp'] for v in self.cache.values()]) if self.cache else None,
            'newest_entry': max([v['timestamp'] for v in self.cache.values()]) if self.cache else None
        }
