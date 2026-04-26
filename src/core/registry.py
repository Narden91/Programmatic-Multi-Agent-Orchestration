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
                    atom_embedding TEXT DEFAULT '[]',
                    payload TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(script_id) REFERENCES scripts(id) ON DELETE CASCADE
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS atom_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    script_id INTEGER NOT NULL,
                    source_atom_id TEXT NOT NULL,
                    target_atom_id TEXT NOT NULL,
                    edge_type TEXT DEFAULT 'dependency',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(script_id) REFERENCES scripts(id) ON DELETE CASCADE
                )
            ''')
            self._ensure_column(cursor, "scripts", "metadata", "TEXT DEFAULT '{}' ")
            self._ensure_column(cursor, "script_atoms", "atom_embedding", "TEXT DEFAULT '[]'")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_script_atoms_script_id ON script_atoms(script_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_script_atoms_agent_type ON script_atoms(agent_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_atom_edges_script_id ON atom_edges(script_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_atom_edges_source ON atom_edges(source_atom_id)")
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
        if not text:
            return []
        if not self.model:
            return []  # Fallback if sentence-transformers not available
        vector = self.model.encode(text)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        return list(vector)

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.model:
            return [[] for _ in texts]

        vectors = self.model.encode(texts)
        if hasattr(vectors, "tolist"):
            vectors = vectors.tolist()
        return [list(vector) for vector in vectors]

    @staticmethod
    def _vector_cosine_similarity(query_emb: List[float], embeddings: List[List[float]]) -> List[float]:
        if not query_emb or not embeddings:
            return []

        try:
            import numpy as np

            matrix = np.asarray(embeddings, dtype=float)
            query_vec = np.asarray(query_emb, dtype=float)
            if matrix.ndim != 2 or query_vec.ndim != 1 or matrix.shape[1] != query_vec.shape[0]:
                raise ValueError("shape mismatch")

            numerators = matrix @ query_vec
            denominators = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
            safe_denominators = np.where(denominators == 0, 1e-12, denominators)
            return (numerators / safe_denominators).tolist()
        except Exception:
            import math

            query_norm = math.sqrt(sum(value * value for value in query_emb))
            scores: List[float] = []
            for embedding in embeddings:
                if len(embedding) != len(query_emb):
                    scores.append(0.0)
                    continue
                numerator = sum(left * right for left, right in zip(embedding, query_emb))
                embedding_norm = math.sqrt(sum(value * value for value in embedding))
                denominator = embedding_norm * query_norm
                scores.append((numerator / denominator) if denominator else 0.0)
            return scores

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

    def _store_atom_payloads(
        self,
        cursor: sqlite3.Cursor,
        script_id: int,
        atom_payloads: List[Dict[str, Any]],
    ) -> None:
        normalized_rows: List[Dict[str, Any]] = []
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

            atom_text = str(payload.get("text") or payload.get("compressed_text") or "").strip()
            normalized_rows.append({
                "span_name": str(atom_payload.get("span_name") or ""),
                "agent_type": str(atom_payload.get("agent_type") or ""),
                "response_format": str(atom_payload.get("response_format") or "plain_text"),
                "atom_index": int(atom_payload.get("atom_index") or 0),
                "payload": payload,
                "dependencies": dependencies,
                "evidence_tags": evidence_tags,
                "atom_text": atom_text,
            })

        embeddings = self._get_embeddings([row["atom_text"] for row in normalized_rows])

        for index, row in enumerate(normalized_rows):
            payload = row["payload"]

            try:
                confidence = float(payload.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0

            embedding_json = json.dumps(embeddings[index] if index < len(embeddings) else [])

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
                    atom_embedding,
                    payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                script_id,
                row["span_name"],
                row["agent_type"],
                row["response_format"],
                row["atom_index"],
                str(payload.get("atom_id") or ""),
                str(payload.get("content_hash") or ""),
                confidence,
                json.dumps(row["dependencies"]),
                json.dumps(row["evidence_tags"]),
                embedding_json,
                json.dumps(payload),
            ))

        self._store_atom_edges(cursor, script_id, normalized_rows)

    @staticmethod
    def _store_atom_edges(
        cursor: sqlite3.Cursor,
        script_id: int,
        normalized_rows: List[Dict[str, Any]],
    ) -> None:
        for row in normalized_rows:
            payload = row["payload"]
            source_atom_id = str(payload.get("atom_id") or "").strip()
            if not source_atom_id:
                continue

            for dependency in row["dependencies"]:
                target_atom_id = str(dependency).strip()
                if not target_atom_id:
                    continue
                cursor.execute('''
                    INSERT INTO atom_edges (script_id, source_atom_id, target_atom_id, edge_type)
                    VALUES (?, ?, ?, ?)
                ''', (script_id, source_atom_id, target_atom_id, "dependency"))

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

    def get_atom_edges(self, script_id: int) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT source_atom_id, target_atom_id, edge_type
                FROM atom_edges
                WHERE script_id = ?
                ORDER BY id ASC
            ''', (script_id,))
            rows = cursor.fetchall()
        finally:
            conn.close()

        return [
            {
                "source_atom_id": source_atom_id,
                "target_atom_id": target_atom_id,
                "edge_type": edge_type,
            }
            for source_atom_id, target_atom_id, edge_type in rows
        ]

    def search_atoms(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []

        query_emb = self._get_embedding(query)
        if not query_emb:
            return []

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    sa.script_id,
                    sa.span_name,
                    sa.agent_type,
                    sa.response_format,
                    sa.atom_index,
                    sa.atom_id,
                    sa.content_hash,
                    sa.confidence,
                    sa.dependencies,
                    sa.evidence_tags,
                    sa.atom_embedding,
                    sa.payload,
                    s.task_description,
                    s.script_content,
                    s.metadata,
                    s.score
                FROM script_atoms sa
                JOIN scripts s ON s.id = sa.script_id
                WHERE sa.atom_embedding IS NOT NULL AND sa.atom_embedding != '[]'
            ''')
            rows = cursor.fetchall()
        finally:
            conn.close()

        embeddings: List[List[float]] = []
        results: List[Dict[str, Any]] = []
        for row in rows:
            (
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
                atom_embedding,
                payload,
                task_description,
                script_content,
                metadata,
                score,
            ) = row

            embedding = json.loads(atom_embedding or "[]")
            if not embedding:
                continue

            embeddings.append(embedding)
            results.append({
                "script_id": script_id,
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
                "task_description": task_description,
                "script_content": script_content,
                "metadata": json.loads(metadata or "{}"),
                "score": score,
            })

        scores = self._vector_cosine_similarity(query_emb, embeddings)
        for result, similarity in zip(results, scores):
            result["similarity"] = similarity

        results.sort(key=lambda item: item.get("similarity", 0.0), reverse=True)
        return results[:top_k]

    def search_atom_neighborhoods(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        neighborhoods: List[Dict[str, Any]] = []
        for seed in self.search_atoms(query, top_k=top_k):
            script_id = int(seed.get("script_id", 0) or 0)
            seed_atom_id = str(seed.get("atom_id") or "").strip()
            if script_id <= 0 or not seed_atom_id:
                continue

            atoms = self.get_script_atoms(script_id)
            edges = self.get_atom_edges(script_id)
            atoms_by_id = {
                str(atom.get("atom_id") or atom.get("payload", {}).get("atom_id") or ""): atom
                for atom in atoms
            }
            related_ids = set()
            related_edges: List[Dict[str, Any]] = []

            for edge in edges:
                source_atom_id = edge["source_atom_id"]
                target_atom_id = edge["target_atom_id"]
                if seed_atom_id not in {source_atom_id, target_atom_id}:
                    continue
                related_edges.append(edge)
                if source_atom_id != seed_atom_id:
                    related_ids.add(source_atom_id)
                if target_atom_id != seed_atom_id:
                    related_ids.add(target_atom_id)

            neighborhoods.append({
                "seed": seed,
                "neighbors": [atoms_by_id[atom_id] for atom_id in related_ids if atom_id in atoms_by_id],
                "edges": related_edges,
            })

        return neighborhoods
        
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
