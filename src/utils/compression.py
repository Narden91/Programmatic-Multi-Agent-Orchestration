"""Entropy-Budgeted Motif Dictionary & Compression Engine for MOSAIC-MoE.

Provides dictionary coding and semantic atom compression for orchestration scripts,
trace structures, and persistent memory graphs.
"""

from __future__ import annotations

import collections
import hashlib
import json
import re
import time
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


_MAGIC_HEADER = b"\x00MOSAIC\x01\x00"

# Standard orchestration building blocks frequently generated across models
DEFAULT_BUILTIN_MOTIFS: Tuple[str, ...] = (
    "async def orchestrate():",
    "await query_agent(",
    'query_agent("technical", ',
    'query_agent("analytical", ',
    'query_agent("creative", ',
    'query_agent("general", ',
    'query_agent("critical-thinker", ',
    "await asyncio.gather(",
    "await memory_store(",
    "await memory_search(",
    "await compress_context(",
    "return ",
    '{"response_format": "semantic_atoms"',
    '"dependencies": []',
    '"evidence_tags": []',
    '"confidence": 1.0',
    '"confidence": 0.9',
    '"confidence": 0.8',
    '"source": "plain_text_fallback"',
)


@dataclass
class CompressionStats:
    """Statistics for a compression or batch compression operation."""

    original_bytes: int = 0
    compressed_bytes: int = 0
    encode_time_ms: float = 0.0
    decode_time_ms: float = 0.0
    count: int = 0

    @property
    def bytes_saved(self) -> int:
        return max(0, self.original_bytes - self.compressed_bytes)

    @property
    def compression_ratio(self) -> float:
        if self.original_bytes <= 0:
            return 1.0
        return self.compressed_bytes / self.original_bytes

    @property
    def space_savings_pct(self) -> float:
        if self.original_bytes <= 0:
            return 0.0
        return (1.0 - self.compression_ratio) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_bytes": self.original_bytes,
            "compressed_bytes": self.compressed_bytes,
            "bytes_saved": self.bytes_saved,
            "compression_ratio": round(self.compression_ratio, 4),
            "space_savings_pct": round(self.space_savings_pct, 2),
            "encode_time_ms": round(self.encode_time_ms, 3),
            "decode_time_ms": round(self.decode_time_ms, 3),
            "count": self.count,
        }


class MotifDictionaryCoder:
    """
    Entropy-budgeted dictionary coder that compresses repetitive prompt/code motifs.
    Combines structured motif substitution with zlib entropy coding.
    """

    def __init__(
        self,
        custom_motifs: Optional[List[str]] = None,
        min_compress_bytes: int = 64,
        zlib_level: int = 6,
    ) -> None:
        self.min_compress_bytes = max(0, int(min_compress_bytes))
        self.zlib_level = min(9, max(1, int(zlib_level)))
        self._motifs: List[str] = []
        self._motif_to_id: Dict[str, int] = {}
        self._id_to_motif: Dict[int, str] = {}
        self._cumulative_stats = CompressionStats()

        initial_motifs = list(DEFAULT_BUILTIN_MOTIFS)
        if custom_motifs:
            for m in custom_motifs:
                if m and m not in initial_motifs:
                    initial_motifs.append(m)

        for motif in initial_motifs:
            self._register_motif(motif)

    def _register_motif(self, motif: str) -> int:
        if motif in self._motif_to_id:
            return self._motif_to_id[motif]

        idx = len(self._motifs)
        self._motifs.append(motif)
        self._motif_to_id[motif] = idx
        self._id_to_motif[idx] = motif
        return idx

    @property
    def dictionary_size(self) -> int:
        return len(self._motifs)

    def get_motifs(self) -> List[str]:
        return list(self._motifs)

    def learn_motifs(
        self,
        texts: List[str],
        max_new_motifs: int = 50,
        min_length: int = 8,
        max_length: int = 60,
        min_frequency: int = 2,
    ) -> List[str]:
        """
        Dynamically discover repeating substrings across a corpus of texts
        and register them in the motif dictionary.
        """
        candidates: collections.Counter[str] = collections.Counter()

        for text in texts:
            if not text or len(text) < min_length:
                continue

            # Tokenize into common code phrases/lines
            lines = text.splitlines()
            for line in lines:
                trimmed = line.strip()
                if min_length <= len(trimmed) <= max_length:
                    candidates[trimmed] += 1

            # Also check common keywords/blocks
            matches = re.findall(r'query_agent\([^)]+\)', text)
            for m in matches:
                if min_length <= len(m) <= max_length:
                    candidates[m] += 1

        discovered: List[str] = []
        for phrase, count in candidates.most_common():
            if count < min_frequency:
                break
            if phrase not in self._motif_to_id:
                self._register_motif(phrase)
                discovered.append(phrase)
                if len(discovered) >= max_new_motifs:
                    break

        return discovered

    def compress(self, text: str) -> bytes:
        """
        Compress text using motif substitution followed by zlib entropy coding.
        Returns a byte payload with `_MAGIC_HEADER`.
        """
        if not isinstance(text, str):
            text = str(text)

        raw_bytes = text.encode("utf-8")
        raw_len = len(raw_bytes)

        if raw_len < self.min_compress_bytes:
            # Below threshold: store plain UTF-8 with uncompressed marker
            payload = _MAGIC_HEADER + b"\x00" + raw_bytes
            self._cumulative_stats.original_bytes += raw_len
            self._cumulative_stats.compressed_bytes += len(payload)
            self._cumulative_stats.count += 1
            return payload

        t0 = time.perf_counter()

        # Phase 1: Motif token replacement (longest motifs first)
        transformed = text
        sorted_motifs = sorted(
            self._motif_to_id.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

        used_motifs: Dict[int, str] = {}
        for motif, m_id in sorted_motifs:
            if motif in transformed:
                placeholder = f"\x1b[{m_id}\x1b"
                transformed = transformed.replace(motif, placeholder)
                used_motifs[m_id] = motif

        # Phase 2: Pack motif map + substituted string
        meta_header = json.dumps(list(used_motifs.keys())).encode("utf-8")
        body = transformed.encode("utf-8")
        combined = len(meta_header).to_bytes(4, "big") + meta_header + body

        # Phase 3: Zlib deflate
        compressed_body = zlib.compress(combined, level=self.zlib_level)
        payload = _MAGIC_HEADER + b"\x01" + compressed_body

        encode_duration = (time.perf_counter() - t0) * 1000.0

        self._cumulative_stats.original_bytes += raw_len
        self._cumulative_stats.compressed_bytes += len(payload)
        self._cumulative_stats.encode_time_ms += encode_duration
        self._cumulative_stats.count += 1

        return payload

    def decompress(self, payload: bytes | str) -> str:
        """
        Decompress a payload previously generated by `compress()`.
        If the payload is plain text or uncompressed, returns it as-is.
        """
        if isinstance(payload, str):
            # Already decompressed or legacy string
            return payload

        if not isinstance(payload, (bytes, bytearray)):
            return str(payload)

        if not payload.startswith(_MAGIC_HEADER):
            # Not a MOSAIC compressed payload; attempt standard UTF-8 decode
            try:
                return payload.decode("utf-8")
            except UnicodeDecodeError:
                return payload.decode("latin1", errors="replace")

        t0 = time.perf_counter()

        flag = payload[len(_MAGIC_HEADER):len(_MAGIC_HEADER) + 1]
        body_bytes = payload[len(_MAGIC_HEADER) + 1:]

        if flag == b"\x00":
            # Stored uncompressed
            return body_bytes.decode("utf-8")

        # Decompress zlib
        decompressed_combined = zlib.decompress(body_bytes)
        header_len = int.from_bytes(decompressed_combined[:4], "big")
        meta_json = decompressed_combined[4:4 + header_len].decode("utf-8")
        body_text = decompressed_combined[4 + header_len:].decode("utf-8")

        motif_ids: List[int] = json.loads(meta_json)
        reconstructed = body_text
        for m_id in motif_ids:
            motif = self._id_to_motif.get(m_id)
            if motif:
                placeholder = f"\x1b[{m_id}\x1b"
                reconstructed = reconstructed.replace(placeholder, motif)

        decode_duration = (time.perf_counter() - t0) * 1000.0
        self._cumulative_stats.decode_time_ms += decode_duration

        return reconstructed

    def get_stats(self) -> CompressionStats:
        return self._cumulative_stats

    def reset_stats(self) -> None:
        self._cumulative_stats = CompressionStats()


_default_coder: Optional[MotifDictionaryCoder] = None


def get_default_coder() -> MotifDictionaryCoder:
    """Global singleton instance of MotifDictionaryCoder."""
    global _default_coder
    if _default_coder is None:
        _default_coder = MotifDictionaryCoder()
    return _default_coder
