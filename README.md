<div align="center">

# 🧠 Programmatic Multi-Agent Orchestration

**A Code-Driven Mixture of Experts (MoE) Architecture powered by LangGraph**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-≥0.2.0-green)](https://github.com/langchain-ai/langgraph)
[![Groq](https://img.shields.io/badge/Groq-Fast_LLM-orange)](https://groq.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-optional-lightgrey)](https://openai.com)
[![Anthropic](https://img.shields.io/badge/Anthropic-optional-lightgrey)](https://anthropic.com)
[![React](https://img.shields.io/badge/React-UI-61DAFB)](https://react.dev)
[![Tests](https://img.shields.io/badge/tests-144%20collected-brightgreen)]()
[![Version](https://img.shields.io/badge/version-0.5.0-blue)]()

*Stop writing static DAGs. Let the AI write its own multi-agent execution graphs on the fly.*

</div>

---

## ✨ The Paradigm Shift: Code-as-Orchestration

Traditional multi-agent frameworks rely on developer-defined, static Directed Acyclic Graphs (DAGs). This project instead treats orchestration itself as generated code.

When a query arrives, the **Master Orchestrator** retrieves prior successful scripts, semantic atoms, dependency neighborhoods, and plan motifs from a persistent registry, then writes one or more **async Python scripts** to solve the task. The selected script runs in a hardened sandbox, awaits expert calls through a unified `query_agent(...)` tool contract, reasons over intermediate `AgentResult` objects in native Python, and feeds the outcome back into the registry for future reuse.

### 🌟 Key Features

| Category | Feature |
|----------|---------|
| **Core** | 🧩 Dynamic `async def orchestrate()` generation per query |
| **Core** | 🎯 Candidate search with heuristic pre-selection, retry repair, and graph-aware mode diversification |
| **Core** | 🤖 Unified `query_agent(agent_type, prompt)` contract returning `AgentResult` with `.text` and semantic `.atoms` |
| **Core** | ⚡ AST speculative execution pass that can rewrite independent waits into `asyncio.gather(...)` |
| **Multi-Provider** | ⚡ Groq (default), OpenAI, and Anthropic seamlessly auto-detected via `LLMFactory` and dynamically bound |
| **Multi-Provider** | 🔌 Per-expert provider override — mix Groq for speed and OpenAI for depth in the same pipeline |
| **Memory** | 💬 Multi-turn conversation memory with sliding window and optional JSON persistence |
| **Memory** | 📚 Persistent registry storing scripts, `script_atoms`, `atom_edges`, `plan_motifs`, and learning metadata |
| **Memory** | 🕸️ Atom-level few-shot retrieval, dependency neighborhoods, and compressed plan motifs for warm-task reuse |
| **Memory** | ⚡ `numpy`-vectorized O(n) similarity fallback for offline registry search |
| **Learning** | 📈 Learning-ranked retrieval that blends similarity with success rate, retries, token cost, and reuse signals |
| **Observability** | 📊 Streaming trace systems for pipeline events and sandbox span traces (`trace_dna`) |
| **Observability** | 📈 Token tracking with per-model cost estimation and static AST execution-plan analysis |
| **Benchmarks** | 🧪 Standard suite with repeats, family aggregates, selection-bias slices, and warm-task graph-retrieval slices |
| **Extensibility** | 🧩 Dynamic expert registry — add/remove expert types at runtime |
| **Security** | 🔒 `SecretStr` wrapper prevents API keys from leaking in repr/logs/tracebacks |
| **Security** | 🔒 Hardened sandbox with AST validation, restricted builtins, bounded `print()`, and configurable policy limits |
| **Security** | 🔒 Trace event redaction, bounded history, and owner-only persisted memory files |
| **UI** | 💻 React interface with FastAPI backend and real-time orchestration insights |

---

## 🏗️ Architecture

```mermaid
graph TD
    A[User Query + Conversation Context] --> B(Orchestrator Agent)
    K[(Registry: scripts + atoms + motifs + learning)] --> B
    B -->|Generates async Python candidate(s)| C{Hardened Code Sandbox}

    C -->|await query_agent("technical", ...)| D[Technical Agent]
    C -->|await query_agent("analytical", ...)| E[Analytical Agent]
    C -->|await query_agent("creative", ...)| F[Creative Agent]
    C -->|await query_agent("general", ...)| G[General Agent]

    D -.->|AgentResult.text / .atoms| C
    E -.->|AgentResult.text / .atoms| C
    F -.->|AgentResult.text / .atoms| C
    G -.->|AgentResult.text / .atoms| C

    C --> H[CodeExecutionAgent]
    H --> I[LangGraph State]
    I --> J[Trace / Metrics / Sandbox Output]
    I --> K
    I --> Z[User Output]
```

### The Workflow

1. **Input** — A query arrives, enriched with conversation context from the sliding-window memory.
2. **Retrieve** — The orchestrator queries the registry for similar scripts, semantic atoms, atom neighborhoods, and plan motifs.
3. **Generate** — The orchestrator writes one or more `async def orchestrate():` candidates, optionally biasing them toward retrieved dependency structures and scheduling motifs.
4. **Validate** — The sandbox performs AST analysis: imports, dangerous attributes, and blocked builtins are rejected *before* any code runs.
5. **Execute** — The selected script runs inside a restricted `exec` with whitelisted builtins, a bounded `print` sink, tracked `query_agent(...)` handles, and a configurable timeout.
6. **Learn** — Successful runs are scored, then persisted back into the registry along with full atom payloads, dependency edges, plan motifs, and execution metadata.
7. **Observe** — Trace events, token usage, retry counts, and retrieval reuse metrics are attached to the LangGraph state and benchmark harness.

---

## 📁 Project Structure

```
.github/
├── AGENTS.md              # Operational handoff for coding agents
api/
├── main.py                # FastAPI app entrypoint
├── routes.py              # /health, /init, /query endpoints
└── schemas.py             # Request/response models
benchmarks/
├── run.py                 # Benchmark CLI with slice modes plus JSON/plot export
├── plotting.py            # Benchmark JSON export helpers and comparison plots
└── suite.py               # BenchmarkSuite, reports, standard cases
frontend/
└── src/                   # React dashboard
src/
├── agents/
│   ├── base.py            # Shared retry behavior
│   ├── orchestrator.py    # Orchestrator + CodeExecution agents
│   ├── registry.py        # Dynamic expert registry
│   ├── tools.py           # Expert spawning + runtime registration helpers
│   └── experts/
│       └── generic.py     # Generic expert implementation
├── core/
│   ├── agents.py          # `query_agent` contract, AgentResult, SemanticAtom
│   ├── config.py          # MoEConfig, SecretStr, LLMConfig, ExpertConfig
│   ├── memory.py          # Ephemeral in-sandbox semantic memory
│   ├── registry.py        # Script/atom/motif persistence and retrieval
│   ├── sandbox.py         # Hardened sandbox with AST validation
│   ├── scoring.py         # Execution-quality scoring
│   ├── state.py           # LangGraph state schema
│   └── tracing.py         # Sandbox span tracing
├── graph/
│   └── builder.py         # LangGraph workflow builder
├── llm/
│   ├── prompts.py         # Prompt templates and retry contract
│   └── providers.py       # LLMFactory: Groq, OpenAI, Anthropic
└── utils/
    ├── cache.py           # LRU + TTL response cache
    ├── code_analyzer.py   # AST-based execution plan extraction
    ├── embeddings.py      # Embedding model loader
    ├── logging.py         # Structured logging
    ├── memory.py          # Conversation memory (sliding window + persistence)
    ├── metrics.py         # Token tracking & cost estimation
    ├── script_bank.py     # Legacy script bank retained for compatibility
    └── tracing.py         # Pipeline event tracing
tests/                     # 144 collected tests across unit, integration, and benchmark slices
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- [UV](https://github.com/astral-sh/uv) (recommended) or pip
- At least one LLM API key:
  - [Groq](https://console.groq.com) (free tier available — recommended)
  - [OpenAI](https://platform.openai.com) (optional)
  - [Anthropic](https://console.anthropic.com) (optional)

### Installation

```bash
# Clone the repository
git clone https://github.com/Narden91/Programmatic-Multi-Agent-Orchestration.git
cd Programmatic-Multi-Agent-Orchestration

# Install with UV (recommended)
uv sync

# Or with pip
pip install -e .

# Optional: install extra providers
pip install -e ".[openai]"       # OpenAI support
pip install -e ".[anthropic]"    # Anthropic support
pip install -e ".[all-providers]" # Both
```

### Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here

# Optional — enable multi-provider support
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

### Run the App

#### Windows Native (Recommended)

Start the entire stack (Vite + Uvicorn) seamlessly using the provided PowerShell script. It automatically manages dependencies, builds the frontend if needed, handles WSL artifact cleanup (preventing `Access Denied` errors), and totally suppresses standard PyTorch/Transformers downloading console spam.

```powershell
# from repository root
.\start.ps1
```

*(Note: Use `.\start.ps1 -Build` to force a production frontend compilation before booting).*

#### Manual startup (all platforms)

```bash
# Terminal 1 (backend)
uv run uvicorn api.main:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 (frontend)
cd frontend
npm install
npm run dev
```

Frontend: `http://127.0.0.1:5173`  
Backend health: `http://127.0.0.1:8000/api/health`

> Important: if you see `http proxy error: /api/init ECONNREFUSED 127.0.0.1:8000`, the frontend is running but the backend is not reachable. Start the backend first and re-open the frontend.

### Recent Updates (Apr 2026)

- **Semantic Memory Graph**: Successful runs now persist full semantic atom payloads, dependency edges, compressed plan motifs, and learning aggregates into the registry.
- **Graph-Shaped Retrieval**: The orchestrator now uses atom-level few-shot hints, dependency neighborhoods, plan motifs, and metadata-biased candidate search instead of relying on script-level retrieval alone.
- **Evaluation Slices**: The benchmark CLI now supports `--selection-bias-slice`, `--warm-task-slice`, repeats/family aggregates, and `--model` overrides that apply to both orchestrator and expert calls.
- **Benchmark Visuals**: `benchmarks.run` can now emit JSON summaries and save comparison plots for before/after evaluation slices.
- **Sandbox Contract Hardening**: `query_agent(...)` handles are task-compatible for scheduling patterns, and the prompt contract now explicitly warns against accessing `.text` or `.atoms` before `await`.

---

## 💻 Programmatic Usage

```python
import asyncio
from src.core.config import MoEConfig, SecretStr
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder

async def main():
    # 1. Configure (keys are wrapped in SecretStr to prevent leakage)
    config = MoEConfig(groq_api_key=SecretStr("your_key"))
    graph = MoEGraphBuilder(config).build()

    # 2. Create initial state
    state = create_initial_state(
        "Explain black holes. Compare them to an everyday object, "
        "then give the physics."
    )

    # 3. Execute the graph
    result = await graph.ainvoke(state)

    # 4. View results
    print("--- Generated Orchestration Code ---")
    print(result["generated_code"])

    print("\n--- Final Answer ---")
    print(result["final_answer"])

if __name__ == "__main__":
    asyncio.run(main())
```

### Multi-Provider Configuration

```python
from src.core.config import MoEConfig, SecretStr, ExpertConfig, LLMConfig

config = MoEConfig(
    groq_api_key=SecretStr("gsk_..."),
    openai_api_key=SecretStr("sk-..."),
    expert_configs={
        "technical": ExpertConfig(
            name="technical",
            description="Programming, technology, sciences",
            llm_config=LLMConfig(model_name="gpt-4o"),
            system_prompt="You are a technical expert.",
            provider_type="openai",          # ← this expert uses OpenAI
        ),
        "creative": ExpertConfig(
            name="creative",
            description="Storytelling, brainstorming",
            llm_config=LLMConfig(model_name="llama-3.3-70b-versatile"),
            system_prompt="You are a creative expert.",
            provider_type="groq",            # ← this one uses Groq
        ),
    },
)
```

### Dynamic Expert Registration

```python
from src.agents.tools import register_expert

register_expert(
    expert_type="legal",
    description="Contract law, compliance, regulation",
    prompt_template='You are a legal expert.\n\nQuery: "{query}"\n\nRespond:',
    system_prompt="You are a legal expert.",
)

# The "legal" expert is now available to generated sandbox code as:
# result = await query_agent("legal", "Review this clause")
```

### Streaming Traces

```python
from src.utils.tracing import get_tracer

tracer = get_tracer()

async for event in tracer.subscribe():
    print(f"[{event.kind}] {event.agent}: {event.data}")
```

### Conversation Memory

```python
from src.utils.memory import ConversationMemory

mem = ConversationMemory(max_turns=10, persist_path="history.json")
mem.add("What is Python?", "Python is a programming language…")
context = mem.format_context()  # inject into prompts for follow-up awareness
```

---

### Groq (default)

| Model | Notes |
|-------|-------|
| `llama-3.1-8b-instant` | Default for orchestrator & experts; best quota-aware starting point |
| `llama-3.3-70b-versatile` | Higher quality, but more likely to hit Groq quota limits |
| `llama3-8b-8192` | Extremely fast |
| `mixtral-8x7b-32768` | Balanced performance |
| `gemma2-9b-it` | Efficient, high quality |

### OpenAI (optional, requires `pip install -e ".[openai]"`)

| Model | Notes |
|-------|-------|
| `gpt-4o` | State of the art multimodal |
| `gpt-4o-mini` | Super fast and cheap |

### Anthropic (optional, requires `pip install -e ".[anthropic]"`)

| Model | Notes |
|-------|-------|
| `claude-3-5-sonnet-20240620` | Highest intelligence |
| `claude-3-5-haiku-20241022` | Fastest Claude model |

### Custom Providers

```python
from src.llm.providers import LLMFactory, LLMProvider

class MyProvider(LLMProvider):
    provider_name = "my_provider"
    def invoke(self, prompt): ...
    async def ainvoke(self, prompt): ...

LLMFactory.register_provider("my_provider", MyProvider)
```

---

## 🔒 Security

The system implements defence-in-depth across multiple layers:

| Layer | Protection |
|-------|-----------|
| **AST validation** | Imports, `__globals__`, `__builtins__`, `eval`, `exec`, `open`, `getattr`, and 20+ dangerous constructs are rejected *before* execution |
| **Restricted builtins** | Only a curated whitelist of safe builtins is exposed inside the sandbox |
| **Bounded stdout** | `print()` is replaced with a capped buffer (`_SandboxPrinter`, 10 KB limit) — no real stdout access |
| **Execution timeout** | `asyncio.wait_for` enforces configurable wall-clock limits (default 120 s) |
| **Secret protection** | API keys are wrapped in `SecretStr` — masked in `repr()`, `str()`, logs, and tracebacks |
| **Trace redaction** | User queries are excluded from trace events; history is bounded (default 500 entries) |
| **Error surface** | Full tracebacks stay server-side; API callers receive the exception message without the traceback |
| **File permissions** | Persisted conversation files are written with `0o600` (owner-only) permissions |
| **No shell injection** | CLI uses `subprocess.run` with explicit argument lists — no `os.system` |
| **Cache integrity** | SHA-256 for cache key generation |

---

## 🧪 Testing

```bash
# Run the full test suite (currently 144 collected tests)
python -m pytest tests/ -v

# With coverage report
python -m pytest tests/ --cov=src --cov-report=html

# Run only unit tests
python -m pytest tests/test_agents.py tests/test_graph.py tests/test_orchestrator.py -v

# Run integration tests (requires GROQ_API_KEY)
python -m pytest tests/test_integration.py tests/test_groq.py -v
```

### Benchmarks

```bash
# Run the standard benchmark suite (requires GROQ_API_KEY)
python -m benchmarks.run

# Run each benchmark case 5 times and print per-case aggregates
python -m benchmarks.run --repeats 5

# Compare baseline retrieval vs metadata-biased candidate selection
python -m benchmarks.run --repeats 5 --selection-bias-slice

# Compare baseline retrieval vs graph-aware warm-task retrieval
python -m benchmarks.run --repeats 5 --warm-task-slice

# Use a smaller model for quota-aware smoke slices (applies to orchestrator and experts)
python -m benchmarks.run --model llama-3.1-8b-instant --filter single_technical --selection-bias-slice

# Save a before/after slice as JSON plus a comparison figure
python -m benchmarks.run --model llama-3.1-8b-instant --repeats 5 --selection-bias-slice --output-json artifacts/selection_bias.json --plot-output artifacts/selection_bias.png

# Re-render a saved benchmark JSON into a plot with a custom title
python -m benchmarks.plotting artifacts/selection_bias.json --output artifacts/selection_bias.pdf --title "Selection-Bias Slice"
```

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (required unless another provider is configured) |
| `OPENAI_API_KEY` | — | OpenAI API key (optional) |
| `ANTHROPIC_API_KEY` | — | Anthropic API key (optional) |
| `ORCHESTRATOR_MODEL` | `llama-3.1-8b-instant` | Model for the orchestrator agent |
| `MAX_TOKENS` | `2000` | Maximum tokens per LLM call |
| `MAX_PARALLEL_EXPERTS` | `4` | Max concurrent expert calls |
| `REQUEST_TIMEOUT` | `120` | Sandbox execution timeout (seconds) |
| `MAX_RETRIES` | `3` | LLM call retry attempts |
| `ORCHESTRATOR_CANDIDATES` | `1` | Number of candidate scripts generated per request; values >1 enable heuristic pre-selection |
| `ORCHESTRATOR_SCRIPT_FEW_SHOTS` | `2` | Number of script-level few-shot examples retrieved for the orchestrator prompt |
| `ORCHESTRATOR_ATOM_FEW_SHOTS` | `4` | Number of atom-level few-shot hints retrieved from `script_atoms` |
| `ENABLE_ATOM_FEW_SHOT_RETRIEVAL` | `true` | Enable/disable atom-level few-shot prompt hints from the registry |
| `ENABLE_METADATA_SELECTION_BIAS` | `true` | Enable/disable metadata-aware candidate ranking using prior atom-rich parallel scripts |
| `REGISTRY_DB_PATH` | `.moe_registry.db` | SQLite registry used for scripts, atoms, motifs, and learning metadata |
| `SANDBOX_ISOLATE_PROCESS` | `false` on Windows / `true` elsewhere | Whether to run the sandbox in a separate process |
| `SANDBOX_MAX_CODE_CHARS` | `30000` | Maximum generated code size accepted by the sandbox |
| `SANDBOX_MAX_AST_NODES` | `8000` | Maximum AST node count allowed before execution |
| `SANDBOX_MAX_STATEMENTS` | `1500` | Maximum statement count allowed before execution |
| `SANDBOX_MAX_QUERY_CALLS` | `120` | Maximum number of `query_agent(...)` calls allowed in a script |
| `ENABLE_CACHE` | `true` | Enable/disable response caching |
| `CACHE_TTL_SECONDS` | `3600` | Cache entry time-to-live |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `DEBUG` | `false` | Enable debug mode |

`LLMConfig.from_env()` also supports per-role overrides such as `TECHNICAL_MODEL`, `ANALYTICAL_MODEL`, `CREATIVE_MODEL`, `GENERAL_MODEL`, and `CRITICAL_THINKER_MODEL`.

### Verify API key loading

1. Create `.env` in repository root with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

2. Start backend and check startup logs for:
    - `dotenv loaded from .../.env`
    - `GROQ_API_KEY detected: True`

3. Verify from browser or terminal:

```bash
curl http://127.0.0.1:8000/api/init
```

Expected JSON contains:

```json
{"has_env_api_key": true, "version": "0.5.0", "models": [...]}
```

---

## 📄 License & Acknowledgments

This project is licensed under the MIT License.

Built with [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), [FastAPI](https://fastapi.tiangolo.com), and [React](https://react.dev).
Special thanks to the open-source AI engineering community.