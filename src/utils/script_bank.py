"""
Script memory bank — stores successful orchestration scripts for few-shot prompting.

When *persist_path* is given the bank survives across process restarts.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class ScriptRecord:
    """A single orchestration script with metadata."""

    query: str
    code: str
    experts_used: List[str]
    success: bool
    timestamp: float = field(default_factory=time.time)

    def _keywords(self) -> set[str]:
        return set(re.findall(r"\w{3,}", self.query.lower()))


class ScriptBank:
    """In-memory bank of orchestration scripts with optional JSON persistence.

    When *persist_path* is given, records are automatically saved / loaded
    from a JSON file so that few-shot examples survive across process
    restarts.
    """

    def __init__(self, persist_path: Optional[str] = None, max_size: int = 50) -> None:
        self._records: List[ScriptRecord] = []
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_size = max_size
        if self._persist_path and self._persist_path.exists():
            self._load()

    # ---- Mutation -------------------------------------------------------

    def record(
        self,
        query: str,
        code: str,
        experts_used: List[str],
        success: bool,
    ) -> None:
        """Append a new script record and optionally persist to disk."""
        self._records.append(
            ScriptRecord(
                query=query,
                code=code,
                experts_used=experts_used,
                success=success,
            )
        )
        if len(self._records) > self._max_size:
            self._records = self._records[-self._max_size :]
        if self._persist_path:
            self._save()

    # ---- Retrieval ------------------------------------------------------

    def find_similar(
        self, query: str, top_k: int = 2, only_success: bool = True
    ) -> List[ScriptRecord]:
        """Return the *top_k* most similar records (keyword-overlap similarity)."""
        target_kw = set(re.findall(r"\w{3,}", query.lower()))
        if not target_kw:
            return []

        candidates = [r for r in self._records if (not only_success or r.success)]
        scored: list[tuple[float, ScriptRecord]] = []
        for r in candidates:
            kw = r._keywords()
            overlap = len(target_kw & kw)
            if overlap > 0:
                score = overlap / max(len(target_kw), len(kw))
                scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    @property
    def size(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()
        if self._persist_path:
            self._save()

    # ---- Persistence ----------------------------------------------------

    def _save(self) -> None:
        data = [asdict(r) for r in self._records]
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        self._persist_path.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text())
            self._records = [ScriptRecord(**d) for d in data]
        except (json.JSONDecodeError, KeyError):
            self._records = []
