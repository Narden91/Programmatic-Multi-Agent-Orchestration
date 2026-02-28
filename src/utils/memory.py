"""Multi-turn conversation memory for the MoE pipeline.

Maintains a sliding-window history of past (query, answer) turns so the
orchestrator can produce context-aware follow-up scripts.

Usage::

    from src.utils.memory import ConversationMemory

    mem = ConversationMemory(max_turns=10)
    mem.add("What is Python?", "Python is a programming language …")
    context = mem.format_context()   # ready to inject into prompts
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    """A single conversation turn."""

    query: str
    answer: str
    experts_used: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "experts_used": self.experts_used,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Turn":
        return cls(
            query=data["query"],
            answer=data["answer"],
            experts_used=data.get("experts_used", []),
            timestamp=data.get("timestamp", 0.0),
        )


class ConversationMemory:
    """Sliding-window conversation memory.

    Parameters
    ----------
    max_turns:
        Maximum number of turns to retain.  Oldest turns are dropped first.
    persist_path:
        Optional JSON file to persist the conversation across sessions.
    """

    def __init__(
        self,
        max_turns: int = 20,
        persist_path: Optional[str | Path] = None,
    ) -> None:
        self.max_turns = max_turns
        self._turns: List[Turn] = []
        self._persist_path = Path(persist_path) if persist_path else None
        if self._persist_path and self._persist_path.exists():
            self._load()

    # ---- Mutation API ------------------------------------------------

    def add(
        self,
        query: str,
        answer: str,
        experts_used: Optional[List[str]] = None,
    ) -> None:
        """Append a turn and enforce the window size."""
        self._turns.append(Turn(
            query=query,
            answer=answer,
            experts_used=experts_used or [],
        ))
        if len(self._turns) > self.max_turns:
            self._turns = self._turns[-self.max_turns:]
        if self._persist_path:
            self._save()

    def clear(self) -> None:
        """Wipe conversation history."""
        self._turns.clear()
        if self._persist_path:
            self._save()

    # ---- Query API ---------------------------------------------------

    @property
    def turns(self) -> List[Turn]:
        return list(self._turns)

    @property
    def last_turn(self) -> Optional[Turn]:
        return self._turns[-1] if self._turns else None

    def __len__(self) -> int:
        return len(self._turns)

    # ---- Prompt formatting -------------------------------------------

    def format_context(self, max_chars: int = 4000) -> str:
        """Build a textual summary of past turns for prompt injection.

        Returns an empty string when the memory is empty, so callers can
        safely include it without branching.
        """
        if not self._turns:
            return ""

        lines: List[str] = ["## Conversation History"]
        total = 0
        # Walk backwards so the most recent turns get priority
        for turn in reversed(self._turns):
            block = (
                f"**User:** {turn.query}\n"
                f"**Answer:** {turn.answer}\n"
            )
            if total + len(block) > max_chars:
                break
            lines.append(block)
            total += len(block)

        # Reverse back to chronological order
        return "\n".join(lines[:1] + lines[1:][::-1])

    # ---- Persistence -------------------------------------------------

    def _save(self) -> None:
        if self._persist_path is None:
            return
        data = [t.to_dict() for t in self._turns]
        self._persist_path.write_text(json.dumps(data, indent=2))
        # Restrict file permissions to owner only (best-effort on Windows)
        try:
            self._persist_path.chmod(0o600)
        except OSError:
            pass

    def _load(self) -> None:
        if self._persist_path is None:
            return
        try:
            raw = json.loads(self._persist_path.read_text())
            self._turns = [Turn.from_dict(d) for d in raw]
        except (json.JSONDecodeError, KeyError):
            self._turns = []
