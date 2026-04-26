import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.utils.embeddings import get_embedding_model

class OrchestrationRegistry:
    """
    Registry for storing and retrieving successful orchestration scripts.
    Uses SQLite to store scripts and their metadata, and sentence-transformers
    to enable semantic search over script goals/descriptions.
    """
    
    def __init__(self, db_path: str = ".moe_registry.db", model_name: str = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self._init_db()
        self.model = get_embedding_model(model_name)
        
    def _init_db(self):
        """Initialize the SQLite database schema for the registry."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_description TEXT NOT NULL,
                    script_content TEXT NOT NULL,
                    embedding TEXT,  -- JSON serialized list of floats
                    metadata TEXT DEFAULT '{}',
                    score REAL DEFAULT 0.0,
                    execution_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS script_atoms (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    script_id INTEGER NOT NULL,
                    span_name TEXT,
                    agent_type TEXT,
                    response_format TEXT,
                    atom_index INTEGER NOT NULL,
                    atom_id TEXT,
                    content_hash TEXT,
                    confidence REAL DEFAULT 0.0,
                    dependencies TEXT DEFAULT '[]',
                    evidence_tags TEXT DEFAULT '[]',
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(script_id) REFERENCES scripts(id) ON DELETE CASCADE
                )
            ''')
            self._ensure_column(cursor, "scripts", "metadata", "TEXT DEFAULT '{}' ")
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _ensure_column(cursor: sqlite3.Cursor, table_name: str, column_name: str, column_sql: str) -> None:
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = {row[1] for row in cursor.fetchall()}
        if column_name not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")

    def _get_embedding(self, text: str) -> List[float]:
        """Compute the embedding for the given text using sentence-transformers."""
        if not self.model:
            return []  # Fallback if sentence-transformers not available
        return self.model.encode(text).tolist()

    def store_script(
        self,
        task_description: str,
        script_content: str,
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        atom_payloads: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        """
        Store a successful script in the registry.
        
        Args:
            task_description: The goal or query that prompted this script.
            script_content: The python source code of the orchestrator script.
            score: The initial quality score of the script.
        """
        embedding = self._get_embedding(task_description)
        embedding_json = json.dumps(embedding)
        
        metadata_json = json.dumps(metadata or {})

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO scripts (task_description, script_content, embedding, metadata, score)
                VALUES (?, ?, ?, ?, ?)
            ''', (task_description, script_content, embedding_json, metadata_json, score))
            script_id = cursor.lastrowid
            self._store_atom_payloads(cursor, script_id, atom_payloads or [])
            conn.commit()
            return script_id
        finally:
            conn.close()

    @staticmethod
    def _store_atom_payloads(
        cursor: sqlite3.Cursor,
        script_id: int,
        atom_payloads: List[Dict[str, Any]],
    ) -> None:
        for atom_payload in atom_payloads:
            if not isinstance(atom_payload, dict):
                continue

            payload = atom_payload.get("payload")
            if not isinstance(payload, dict):
                continue

            dependencies = payload.get("dependencies") or []
            if not isinstance(dependencies, list):
                dependencies = []

            evidence_tags = payload.get("evidence_tags") or []
            if not isinstance(evidence_tags, list):
                evidence_tags = []

            try:
                confidence = float(payload.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0

            cursor.execute('''
                INSERT INTO script_atoms (
                    script_id,
                    span_name,
                    agent_type,
                    response_format,
                    atom_index,
                    atom_id,
                    content_hash,
                    confidence,
                    dependencies,
                    evidence_tags,
                    payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                script_id,
                str(atom_payload.get("span_name") or ""),
                str(atom_payload.get("agent_type") or ""),
                str(atom_payload.get("response_format") or "plain_text"),
                int(atom_payload.get("atom_index") or 0),
                str(payload.get("atom_id") or ""),
                str(payload.get("content_hash") or ""),
                confidence,
                json.dumps(dependencies),
                json.dumps(evidence_tags),
                json.dumps(payload),
            ))

    def search(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Find the most similar past scripts using cosine similarity.
        Since SQLite doesn't have native vector search (without extensions),
        we'll do an exact cosine similarity via python if the DB is small, 
        or just fallback to keyword matching/fetching all if it grows.
        """
        if not self.model:
            return []

        query_emb = self._get_embedding(query)
        if not query_emb:
            return []

        import math
        def cosine_similarity(v1: List[float], v2: List[float]) -> float:
            dot = sum(a * b for a, b in zip(v1, v2))
            norm1 = math.sqrt(sum(a * a for a in v1))
            norm2 = math.sqrt(sum(b * b for b in v2))
            if norm1 == 0 or norm2 == 0: return 0.0
            return dot / (norm1 * norm2)

        results = []
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT id, task_description, script_content, embedding, metadata, score FROM scripts")
            
            for row in cursor.fetchall():
                row_id, desc, content, emb_str, metadata_str, score = row
                if not emb_str:
                    continue
                emb = json.loads(emb_str)
                sim = cosine_similarity(query_emb, emb)
                results.append({
                    "id": row_id,
                    "task_description": desc,
                    "script_content": content,
                    "metadata": json.loads(metadata_str or "{}"),
                    "score": score,
                    "similarity": sim
                })
        finally:
            conn.close()
                
        # Sort by similarity
        results.sort(key=lambda x: x["similarity"], reverse=True)
        return results[:top_k]

    def get_script_atoms(self, script_id: int) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT span_name, agent_type, response_format, atom_index, atom_id, content_hash,
                       confidence, dependencies, evidence_tags, payload
                FROM script_atoms
                WHERE script_id = ?
                ORDER BY id ASC
            ''', (script_id,))
            rows = cursor.fetchall()
        finally:
            conn.close()

        atoms: List[Dict[str, Any]] = []
        for row in rows:
            (
                span_name,
                agent_type,
                response_format,
                atom_index,
                atom_id,
                content_hash,
                confidence,
                dependencies,
                evidence_tags,
                payload,
            ) = row
            atoms.append({
                "span_name": span_name,
                "agent_type": agent_type,
                "response_format": response_format,
                "atom_index": atom_index,
                "atom_id": atom_id,
                "content_hash": content_hash,
                "confidence": confidence,
                "dependencies": json.loads(dependencies or "[]"),
                "evidence_tags": json.loads(evidence_tags or "[]"),
                "payload": json.loads(payload or "{}"),
            })
        return atoms
        
    def update_score(self, script_id: int, new_score: float):
        """Update the score and usage metrics of an existing script."""
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE scripts 
                SET score = ?, execution_count = execution_count + 1, last_used_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (new_score, script_id))
            conn.commit()
        finally:
            conn.close()
