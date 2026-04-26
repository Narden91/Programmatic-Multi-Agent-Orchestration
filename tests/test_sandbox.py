import pytest
import asyncio
import json
from src.core.sandbox import CodeSandbox, SandboxSecurityError
from src.core.agents import AgentResult

@pytest.mark.asyncio
async def test_sandbox_security():
    sandbox = CodeSandbox()
    
    # 1. Test __import__ block
    bad_code1 = '''
async def orchestrate():
    import os
    return os.name
'''
    with pytest.raises(SandboxSecurityError):
        await sandbox.execute(bad_code1)

    # 2. Test getattr block
    bad_code2 = '''
async def orchestrate():
    return getattr(str, "__class__")
'''
    with pytest.raises(SandboxSecurityError):
        await sandbox.execute(bad_code2)

@pytest.mark.asyncio
async def test_sandbox_execution():
    sandbox = CodeSandbox()
    good_code = '''
async def orchestrate():
    return "Hello, Sandbox"
'''
    res = await sandbox.execute(good_code)
    assert res['result'] == "Hello, Sandbox"
    assert "trace" in res


@pytest.mark.asyncio
async def test_sandbox_query_agent_exposes_semantic_atoms(monkeypatch):
    async def fake_query_agent(agent_type, prompt, context_ids=None):
        payload = {
            "summary": "Binary search summary.",
            "atoms": [
                {
                    "claim_id": "bs-1",
                    "compressed_text": "Binary search halves the interval.",
                    "confidence": 0.93,
                    "dependencies": [],
                    "evidence_tags": ["algorithm"],
                },
                {
                    "claim_id": "bs-2",
                    "compressed_text": "It requires sorted data.",
                    "confidence": 0.98,
                    "dependencies": ["bs-1"],
                    "evidence_tags": ["precondition"],
                },
            ],
        }
        return AgentResult.from_response_text(
            json.dumps(payload),
            agent_type=agent_type,
        )

    monkeypatch.setattr("src.core.sandbox.real_query_agent", fake_query_agent)

    sandbox = CodeSandbox(isolate_process=False)
    code = '''
async def orchestrate():
    result = await query_agent("technical", "Explain binary search")
    return f"{len(result.atoms)}:{result.atoms[1].dependencies[0]}:{result.text}"
'''

    res = await sandbox.execute(code)

    assert res["result"] == "2:bs-1:Binary search summary."
    assert res["trace"][0]["outputs"]["atom_count"] == 2
    assert res["trace"][0]["outputs"]["response_format"] == "semantic_atoms"
