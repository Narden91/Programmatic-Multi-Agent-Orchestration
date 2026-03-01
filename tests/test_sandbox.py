import pytest
import asyncio
from src.core.sandbox import CodeSandbox, SandboxSecurityError

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
