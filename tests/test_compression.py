"""Unit tests for Entropy-Budgeted Motif Dictionary & Compression Engine."""

import pytest
from src.utils.compression import (
    DEFAULT_BUILTIN_MOTIFS,
    CompressionStats,
    MotifDictionaryCoder,
    get_default_coder,
)


class TestMotifDictionaryCoder:

    def test_default_coder_has_builtin_motifs(self):
        coder = MotifDictionaryCoder()
        assert coder.dictionary_size >= len(DEFAULT_BUILTIN_MOTIFS)
        motifs = coder.get_motifs()
        assert "async def orchestrate():" in motifs
        assert "await query_agent(" in motifs

    def test_custom_motifs_registered(self):
        coder = MotifDictionaryCoder(custom_motifs=["custom_domain_function()", "SPECIAL_PAYLOAD"])
        assert "custom_domain_function()" in coder.get_motifs()
        assert "SPECIAL_PAYLOAD" in coder.get_motifs()

    def test_lossless_compression_and_decompression(self):
        coder = MotifDictionaryCoder(min_compress_bytes=32)
        code = (
            'async def orchestrate():\n'
            '    res1 = await query_agent("technical", "Analyze memory leak")\n'
            '    res2 = await query_agent("analytical", "Compute statistics")\n'
            '    joined = await asyncio.gather(\n'
            '        query_agent("creative", "Synthesize report"),\n'
            '        query_agent("general", "Proofread summary"),\n'
            '    )\n'
            '    await memory_store("analysis", res1.text)\n'
            '    return joined[0].text\n'
        )

        compressed = coder.compress(code)
        assert isinstance(compressed, bytes)
        assert len(compressed) > 0
        assert compressed != code.encode("utf-8")

        decompressed = coder.decompress(compressed)
        assert decompressed == code

    def test_below_threshold_stored_uncompressed(self):
        coder = MotifDictionaryCoder(min_compress_bytes=500)
        short_text = "short text"
        compressed = coder.compress(short_text)
        assert isinstance(compressed, bytes)
        decompressed = coder.decompress(compressed)
        assert decompressed == short_text

    def test_decompress_plain_string(self):
        coder = MotifDictionaryCoder()
        plain = "plain legacy python code"
        assert coder.decompress(plain) == plain

    def test_decompress_uncompressed_bytes(self):
        coder = MotifDictionaryCoder()
        plain_bytes = b"non-mosaic raw bytes"
        assert coder.decompress(plain_bytes) == "non-mosaic raw bytes"

    def test_learn_motifs_discovers_repeating_patterns(self):
        coder = MotifDictionaryCoder()
        corpus = [
            "res = await query_agent('database_specialist', 'query row')\n"
            "validate_database_result(res)\n",
            "res = await query_agent('database_specialist', 'query table')\n"
            "validate_database_result(res)\n",
            "res = await query_agent('database_specialist', 'query schema')\n"
            "validate_database_result(res)\n",
        ]
        discovered = coder.learn_motifs(corpus, min_frequency=2, min_length=10)
        assert any("validate_database_result" in m for m in discovered)

    def test_compression_stats_tracking(self):
        coder = MotifDictionaryCoder(min_compress_bytes=16)
        coder.reset_stats()
        text = "async def orchestrate():\n    await query_agent('technical', 'test prompt with more tokens')\n" * 10
        coder.compress(text)

        stats = coder.get_stats()
        assert stats.count == 1
        assert stats.original_bytes == len(text.encode("utf-8"))
        assert stats.compressed_bytes > 0
        assert stats.bytes_saved > 0
        assert 0.0 < stats.compression_ratio < 1.0
        assert stats.space_savings_pct > 0.0

        d = stats.to_dict()
        assert "compression_ratio" in d
        assert "space_savings_pct" in d
        assert "bytes_saved" in d

    def test_singleton_accessor(self):
        c1 = get_default_coder()
        c2 = get_default_coder()
        assert c1 is c2
