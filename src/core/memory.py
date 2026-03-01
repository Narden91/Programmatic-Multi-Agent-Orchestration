import tempfile
from typing import List, Dict, Any, Optional
import os

try:
    import lancedb
    import pyarrow as pa
    from sentence_transformers import SentenceTransformer
    import logging
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
except ImportError:
    lancedb = None
    pa = None
    SentenceTransformer = None

class EphemeralMemory:
    """
    In-sandbox vector database that exists only for the duration of the script.
    Agents can store outputs, reasoning traces, or data chunks, and retrieve 
    relevant pieces via semantic search. Supports Context Compression.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.temp_dir = tempfile.TemporaryDirectory()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = SentenceTransformer(model_name) if SentenceTransformer else None
        
        if lancedb:
            self.db = lancedb.connect(self.temp_dir.name)
            
            # Create schema lazily when first item is stored, or explicitly here:
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("metadata", pa.string()), # store JSON
                pa.field("vector", pa.list_(pa.float32(), 384)), # 384 is size of all-MiniLM-L6-v2
            ])
            self.table = self.db.create_table("memory", schema=schema)
        else:
            self.db = None
            self.table = None
            self._fallback_memory = []

    def _get_embedding(self, text: str) -> List[float]:
        if not self.model:
            return [0.0] * 384
        return self.model.encode(text).tolist()

    async def store(self, key: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a piece of text (e.g., agent response, extracted entity) in memory."""
        import json
        emb = self._get_embedding(text)
        meta_str = json.dumps(metadata or {})
        
        if self.table is not None:
            self.table.add([{
                "id": key,
                "text": text,
                "metadata": meta_str,
                "vector": emb
            }])
        else:
            self._fallback_memory.append({
                "id": key,
                "text": text,
                "metadata": meta_str,
                "vector": emb
            })
            
        return key

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search memory for the top_k most relevant chunks using semantic search."""
        import json
        emb = self._get_embedding(query)
        
        if self.table is not None:
            results = self.table.search(emb).limit(top_k).to_list()
            return [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "metadata": json.loads(r["metadata"]),
                    "distance": r.get("_distance", 0.0)
                } for r in results
            ]
        else:
            try:
                import numpy as np
                def cosine_similarity(v1, v2):
                    v1_np, v2_np = np.array(v1), np.array(v2)
                    norm1, norm2 = np.linalg.norm(v1_np), np.linalg.norm(v2_np)
                    if norm1 == 0 or norm2 == 0: return 0.0
                    return float(np.dot(v1_np, v2_np) / (norm1 * norm2))
            except ImportError:
                import math
                def cosine_similarity(v1, v2):
                    dot = sum(a * b for a, b in zip(v1, v2))
                    norm1 = math.sqrt(sum(a * a for a in v1))
                    norm2 = math.sqrt(sum(b * b for b in v2))
                    if norm1 == 0 or norm2 == 0: return 0.0
                    return dot / (norm1 * norm2)
                
            scored = []
            for item in self._fallback_memory:
                sim = cosine_similarity(emb, item["vector"])
                scored.append((sim, item))
            
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {
                    "id": item["id"],
                    "text": item["text"],
                    "metadata": json.loads(item["metadata"]),
                    "distance": 1.0 - sim
                } for sim, item in scored[:top_k]
            ]
            
    def cleanup(self):
        """Clean up the ephemeral database."""
        self.temp_dir.cleanup()

    async def compress_context(self, query: str, top_k: int = 5) -> str:
        """
        Advanced Semantic Context Compression:
        Instead of returning all text chunks and inflating the LLM window,
        this extracts and summarizes the memory based on the query.
        (Currently returns the raw concatenated top chunks, but prepares for LLM summary.)
        """
        results = await self.search(query, top_k)
        if not results:
            return "No relevant context found in memory."
            
        # In a full implementation, we could spawn a 'synthesizer' agent here to summarize.
        compressed = "\n...\n".join([f"[{r['id']}] {r['text']}" for r in results])
        return compressed
