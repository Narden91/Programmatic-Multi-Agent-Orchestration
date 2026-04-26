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
    assert res["trace"][0]["outputs"]["atoms"][1]["dependencies"] == ["bs-1"]
    assert res["trace"][0]["outputs"]["atoms"][0]["atom_id"] == "bs-1"


@pytest.mark.asyncio
async def test_sandbox_cancels_abandoned_query_tasks(monkeypatch):
    cancelled = asyncio.Event()

    async def fake_query_agent(agent_type, prompt, context_ids=None):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.set()
            raise

    monkeypatch.setattr("src.core.sandbox.real_query_agent", fake_query_agent)

    sandbox = CodeSandbox(isolate_process=False)
    code = '''
async def orchestrate():
    pending = asyncio.ensure_future(query_agent("technical", "Explain binary search"))
    await asyncio.sleep(0)
    return 1 / 0
'''

    with pytest.raises(ZeroDivisionError, match="division by zero"):
        await sandbox.execute(code)

    assert cancelled.is_set()


@pytest.mark.asyncio
async def test_tracked_query_handle_exposes_task_methods(monkeypatch):
    async def fake_query_agent(agent_type, prompt, context_ids=None):
        return AgentResult(text=f"ok:{agent_type}")

    monkeypatch.setattr("src.core.sandbox.real_query_agent", fake_query_agent)

    sandbox = CodeSandbox(isolate_process=False)
    code = '''
async def orchestrate():
    pending = query_agent("technical", "Explain binary search")
    before = pending.done()
    result = await pending
    return f"{before}:{pending.done()}:{pending.exception()}:{pending.result().text}:{result.text}"
'''

    res = await sandbox.execute(code)

    assert res["result"] == "False:True:None:ok:technical:ok:technical"
