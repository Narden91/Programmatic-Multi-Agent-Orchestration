"""Integration tests for OrchestrationRegistry with Motif Dictionary Compression."""

import os
import tempfile
import pytest

from src.core.registry import OrchestrationRegistry
from src.utils.compression import MotifDictionaryCoder


class TestCompressionRegistryIntegration:

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        if os.path.exists(db_path):
            os.remove(db_path)

    def test_store_and_search_decompressed(self, temp_db):
        coder = MotifDictionaryCoder(min_compress_bytes=16)
        reg = OrchestrationRegistry(db_path=temp_db, enable_compression=True, coder=coder)

        code = (
            'async def orchestrate():\n'
            '    res = await query_agent("technical", "Explain GIL in Python")\n'
            '    return res.text\n'
        )

        script_id = reg.store_script(
            task_description="Explain GIL in Python",
            script_content=code,
            score=0.9,
            metadata={"selected_experts": ["technical"]},
        )
        assert script_id > 0

        # Retrieve via semantic search
        results = reg.search("Explain GIL in Python", top_k=1)
        assert len(results) == 1
        retrieved_code = results[0]["script_content"]
        assert isinstance(retrieved_code, str)
        assert retrieved_code == code

    def test_motif_learning_persistence_across_instances(self, temp_db):
        coder1 = MotifDictionaryCoder(min_compress_bytes=16)
        reg1 = OrchestrationRegistry(db_path=temp_db, enable_compression=True, coder=coder1)

        custom_code = (
            'async def orchestrate():\n'
            '    # CUSTOM_PIPELINE_HEADER_TAG\n'
            '    a = await query_agent("technical", "Step 1")\n'
            '    return a.text\n'
        )

        reg1.store_script(task_description="Task A", script_content=custom_code)
        reg1.store_script(task_description="Task B", script_content=custom_code)

        # Create fresh registry instance connecting to the same DB
        coder2 = MotifDictionaryCoder(min_compress_bytes=16)
        reg2 = OrchestrationRegistry(db_path=temp_db, enable_compression=True, coder=coder2)

        results = reg2.search("Task A", top_k=1)
        assert len(results) == 1
        assert results[0]["script_content"] == custom_code

    def test_backward_compatibility_with_legacy_uncompressed_scripts(self, temp_db):
        # 1. Store uncompressed script
        reg_plain = OrchestrationRegistry(db_path=temp_db, enable_compression=False)
        plain_code = 'async def orchestrate():\n    return "legacy raw string"\n'
        reg_plain.store_script(task_description="Legacy Query", script_content=plain_code)

        # 2. Open with compression-enabled registry
        reg_compressed = OrchestrationRegistry(db_path=temp_db, enable_compression=True)
        results = reg_compressed.search("Legacy Query", top_k=1)
        assert len(results) == 1
        assert results[0]["script_content"] == plain_code

    def test_compression_stats_reporting(self, temp_db):
        coder = MotifDictionaryCoder(min_compress_bytes=16)
        reg = OrchestrationRegistry(db_path=temp_db, enable_compression=True, coder=coder)

        for i in range(5):
            code = (
                f'async def orchestrate():\n'
                f'    res = await query_agent("technical", "Task iteration {i}")\n'
                f'    return res.text\n'
            )
            reg.store_script(f"Task iteration {i}", code)

        stats = reg.get_compression_stats()
        assert stats["compression_enabled"] is True
        assert stats["dictionary_size"] >= 10
        assert stats["count"] >= 5
        assert stats["original_bytes"] > 0
