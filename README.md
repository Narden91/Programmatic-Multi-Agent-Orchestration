# 🧠 Mixture of Experts with LangGraph

A modular Mixture of Experts (MoE) system powered by LangGraph and Groq LLMs, featuring intelligent query routing and specialized AI agents.

## ✨ Features

- **🧭 Smart Router**: Automatically routes queries to the most suitable expert(s)
- **🤖 Specialized Experts**: Technical, Creative, Analytical, and General knowledge agents
- ** � AI Synthesizer**: Combines multiple expert responses into coherent answers
- **📊 Visual Flow**: Interactive network graphs showing agent decision flow
- **⚡ Fast**: Parallel processing with Groq's high-performance LLMs
- **💬 Chat Interface**: Clean Streamlit UI with example queries and chat history

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- [UV](https://github.com/astral-sh/uv) (recommended) or pip
- Groq API key ([get one free](https://console.groq.com))

### Installation

```bash
# Clone the repository
git clone https://github.com/Narden91/AgentAILangchain.git
cd AgentAILangchain

# Install dependencies with UV (recommended)
uv sync

# Or with pip
pip install -e .
```

### Configuration

Create a `.env` file:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Run

```bash
# With UV
uv run streamlit run ui/app.py

# Or with pip
streamlit run ui/app.py
```

Visit `http://localhost:8501` and start chatting!

## 🤖 Available Models

The system supports the following Groq models:
- **meta-llama/llama-4-maverick-17b-128e-instruct** (default) - 131K context, optimal balance
- **llama-3.3-70b-versatile** - High performance, 128K context
- **qwen/qwen3-32b** - 131K context, 40K output
- **moonshotai/kimi-k2-instruct-0905** - 262K context, long-form content

## � Usage

### Via UI
1. Enter your Groq API key in the sidebar (or load from `.env`)
2. Select your preferred model
3. Click an example query or type your own
4. View the answer, expert analysis, and agent flow visualization

### Programmatic
```python
from src.core.config import MoEConfig
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder

# Initialize
config = MoEConfig(groq_api_key="your_key")
graph = MoEGraphBuilder(config).build()

# Process query
state = create_initial_state("Explain quantum computing")
result = graph.invoke(state)

print(result['final_answer'])
print(result['selected_experts'])
```

## 🏗️ Architecture

```
Query → Router → [Experts] → Synthesizer → Answer
                    ↓
         Technical | Creative
         Analytical | General
```

- **Router**: Analyzes queries and selects relevant experts
- **Experts**: Specialized agents for different domains
- **Synthesizer**: Combines expert insights into final response

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=src
```

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

Built with [LangGraph](https://github.com/langchain-ai/langgraph), [Groq](https://groq.com), and [Streamlit](https://streamlit.io).