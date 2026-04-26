import asyncio
from unittest.mock import AsyncMock

import pytest

from src.agents.orchestrator import CodeExecutionAgent
from src.core.state import create_initial_state


class _CaptureRegistry:
    def __init__(self):
        self.calls = []

    def store_script(self, task_description, script_content, score=0.0, metadata=None, atom_payloads=None):
        self.calls.append({
            "task_description": task_description,
            "script_content": script_content,
            "score": score,
            "metadata": metadata or {},
            "atom_payloads": atom_payloads or [],
        })
        return 1


@pytest.mark.asyncio
async def test_code_execution_agent_stores_trace_summary_metadata():
    agent = CodeExecutionAgent(isolate_process=False)
    capture = _CaptureRegistry()
    agent.orchestration_registry = capture
    agent.scorer.score_execution = AsyncMock(return_value=0.87)

    state = create_initial_state("Explain binary search")
    state["generated_code"] = (
        "async def orchestrate():\n"
        "    result = await query_agent('technical', 'Explain binary search')\n"
        "    return result.text\n"
    )

    agent.sandbox.execute = AsyncMock(return_value={
        "result": "Binary search summary.",
        "selected_experts": ["technical"],
        "expert_responses": {"technical": "Binary search summary."},
        "trace": [
            {
                "type": "agent",
                "name": "query_agent_technical",
                "inputs": {"agent_type": "technical", "prompt": "Explain binary search"},
                "outputs": {
                    "text": "Binary search summary.",
                    "atom_count": 2,
                    "response_format": "semantic_atoms",
                    "atoms": [
                        {
                            "atom_id": "bs-1",
                            "text": "Binary search halves the interval.",
                            "confidence": 0.93,
                            "dependencies": [],
                            "evidence_tags": ["algorithm"],
                            "metadata": {},
                            "content_hash": "abcd1234",
                        },
                        {
                            "atom_id": "bs-2",
                            "text": "It requires sorted data.",
                            "confidence": 0.98,
                            "dependencies": ["bs-1"],
                            "evidence_tags": ["precondition"],
                            "metadata": {},
                            "content_hash": "efgh5678",
                        },
                    ],
                },
            }
        ],
    })

    result = await agent.aexecute(state)

    assert result["final_answer"] == "Binary search summary."
    assert len(capture.calls) == 1
    saved = capture.calls[0]["metadata"]
    assert saved["selected_experts"] == ["technical"]
    assert saved["trace_summary"]["atom_count_total"] == 2
    assert saved["trace_summary"]["response_formats"] == ["semantic_atoms"]
    assert len(capture.calls[0]["atom_payloads"]) == 2
    assert capture.calls[0]["atom_payloads"][1]["payload"]["dependencies"] == ["bs-1"]