import streamlit as st
import sys
from pathlib import Path
import os
import asyncio
import threading
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config import MoEConfig, SecretStr
from src.core.state import create_initial_state
from src.graph.builder import MoEGraphBuilder
from ui.components.visualization import MoEVisualizer

load_dotenv()


st.set_page_config(
    page_title="Mixture of Experts",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("🧠 Mixture of Experts")
st.caption("Intelligent query routing with specialized AI experts")

with st.expander("🚀 Why this is different", expanded=True):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Orchestration", "Dynamic Python", "per query")
    with col_b:
        st.metric("Execution", "Sandboxed", "AST + timeout")
    with col_c:
        st.metric("Experts", "Parallel async", "context-compressed")
    st.markdown(
        "This system does **Code-as-Orchestration**: instead of following a fixed DAG, "
        "the orchestrator writes an async script tailored to your request, executes it in "
        "a hardened sandbox, and returns a synthesized result with transparent execution metadata."
    )


with st.sidebar:
    st.header("⚙️ Configuration")
    
    env_api_key = os.getenv("GROQ_API_KEY", "")
    groq_api_key = st.text_input(
        "Groq API Key",
        value=env_api_key,
        type="password",
        help="Get your API key at https://console.groq.com"
    )
    
    if not groq_api_key:
        st.warning("Please add your Groq API key to continue.")
    
    st.divider()
    
    st.subheader("🤖 Model")
    available_models = [
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "llama-3.3-70b-versatile",
        "qwen/qwen3-32b",
        "moonshotai/kimi-k2-instruct-0905"
    ]
    
    selected_model = st.selectbox(
        "Choose AI Model",
        available_models,
        help="Select the Groq model for all agents"
    )
    
    st.divider()
    
    st.subheader("🤖 Available Experts")
    st.markdown("""
    - 🔧 **Technical**: Code, tech, science
    - 🎨 **Creative**: Stories, ideas, content
    - 📊 **Analytical**: Data, logic, analysis
    - 💬 **General**: Conversations, facts
    """)
    
    st.divider()
    
    with st.expander("ℹ️ How It Works"):
        st.markdown("""
        1. **Orchestrator** analyzes your query
        2. **Writes** an async Python script to solve it
        3. **Sandbox** executes the code securely
        4. **Micro-agents** are invoked programmatically
        5. **Final** answer is returned contextually
        """)
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your Mixture of Experts AI assistant. Ask me anything, and I'll route your question to the most suitable expert(s) for the best answer."
        }
    ]


def render_result_panels(payload: dict, key_prefix: str = ""):
    """Render rich orchestration insights for one assistant response."""
    generated_code = payload.get("generated_code", "")
    selected_experts = payload.get("selected_experts", [])
    expert_responses = payload.get("expert_responses", {})
    token_usage = payload.get("token_usage", {})
    iterations = payload.get("code_execution_iterations", 1)
    code_error = payload.get("code_execution_error", "")

    total_tokens = token_usage.get("total_tokens", 0)
    estimated_cost = token_usage.get("estimated_cost_usd", 0.0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Experts Used", len(selected_experts))
    m2.metric("Iterations", iterations)
    m3.metric("Total Tokens", f"{total_tokens:,}")
    m4.metric("Est. Cost", f"${estimated_cost:.4f}")

    tab_code, tab_flow, tab_plan, tab_tokens, tab_experts = st.tabs(
        ["💻 Code", "🕸️ Agent Flow", "🧩 Execution Plan", "📊 Tokens", "🤖 Expert Outputs"]
    )

    with tab_code:
        st.code(generated_code or "# No code generated", language="python")
        if iterations > 1:
            st.warning(f"Script took {iterations} iterations to execute successfully (auto-retried).")
        if code_error:
            st.error(f"Execution Error: {code_error}")

    with tab_flow:
        if selected_experts:
            flow_fig = MoEVisualizer.create_network_graph(
                selected_experts=selected_experts,
                expert_responses=expert_responses,
                generated_code=generated_code,
            )
            st.plotly_chart(flow_fig, use_container_width=True, key=f"{key_prefix}_flow")
        else:
            st.info("No expert routing data available for this response.")

    with tab_plan:
        if generated_code:
            plan_fig = MoEVisualizer.create_execution_plan_graph(
                code=generated_code,
                actual_experts=selected_experts,
            )
            st.plotly_chart(plan_fig, use_container_width=True, key=f"{key_prefix}_plan")
        else:
            st.info("No execution plan available.")

    with tab_tokens:
        if token_usage:
            token_fig = MoEVisualizer.create_token_usage_chart(token_usage)
            st.plotly_chart(token_fig, use_container_width=True, key=f"{key_prefix}_tokens")
            st.json(token_usage)
        else:
            st.info("Token usage metadata is not available for this response.")

    with tab_experts:
        if expert_responses:
            for expert, response in expert_responses.items():
                with st.expander(f"{expert.title()} Expert", expanded=False):
                    st.markdown(response)
        else:
            st.info("No expert outputs were captured.")


if len(st.session_state.messages) <= 1:
    st.subheader("💡 Try these showcase prompts")
    st.caption("Use these to see dynamic code generation, expert routing, and execution analytics.")
    
    example_queries = [
        {
            "icon": "🔧",
            "text": "Explain quantum computing in simple terms",
            "category": "technical",
            "color": "#fef2f2"
        },
        {
            "icon": "🎨",
            "text": "Write a short story about AI discovering emotions",
            "category": "creative",
            "color": "#f0fdf4"
        },
        {
            "icon": "📊",
            "text": "Compare the pros and cons of remote work",
            "category": "analytical",
            "color": "#eff6ff"
        },
        {
            "icon": "🔧",
            "text": "How do I build a REST API with FastAPI?",
            "category": "technical",
            "color": "#fef2f2"
        },
        {
            "icon": "💬",
            "text": "What are the main causes of climate change?",
            "category": "general",
            "color": "#faf5ff"
        },
        {
            "icon": "🎨",
            "text": "Generate creative marketing slogans for eco-friendly products",
            "category": "creative",
            "color": "#f0fdf4"
        }
    ]
    
    st.markdown(
        """
        <style>
        div.stButton > button {
            border-radius: 12px;
            padding: 0.8rem 1rem;
            text-align: left;
            height: auto;
            white-space: normal;
            line-height: 1.4;
            border: 1px solid #e5e7eb;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    cols = st.columns(2)
    for idx, example in enumerate(example_queries):
        with cols[idx % 2]:
            if st.button(
                example['text'], 
                key=f"example_{idx}",
                use_container_width=True
            ):
                st.session_state.example_to_process = example['text']
                st.rerun()
    
    st.divider()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        if msg["role"] == "assistant" and "generated_code" in msg:
            with st.expander("🔬 Orchestration Insights", expanded=False):
                render_result_panels(msg, key_prefix=f"history_{abs(hash(msg.get('content', '')))}")


def process_query(query: str, api_key: str, model: str):
    """Process a query through the MoE system"""
    try:
        config = MoEConfig(groq_api_key=SecretStr(api_key))
        
        config.router_config.model_name = model
        config.synthesizer_config.model_name = model
        for expert_config in config.expert_configs.values():
            expert_config.llm_config.model_name = model
        
        config.validate()
        
        builder = MoEGraphBuilder(config)
        graph = builder.build()
        
        initial_state = create_initial_state(query)
        timeout_seconds = max(int(config.request_timeout), 1)

        async def _run_graph():
            return await asyncio.wait_for(graph.ainvoke(initial_state), timeout=timeout_seconds)

        def _run_async_sync(coro):
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(coro)

            outcome: dict[str, object] = {}

            def _runner():
                try:
                    outcome["result"] = asyncio.run(coro)
                except Exception as exc:
                    outcome["error"] = exc

            worker = threading.Thread(target=_runner, daemon=True)
            worker.start()
            worker.join()

            if "error" in outcome:
                raise outcome["error"]
            return outcome.get("result")

        result = _run_async_sync(_run_graph())
        
        return result
    except asyncio.TimeoutError:
        st.error(
            f"Request timed out after {timeout_seconds}s. "
            "Try a shorter query or increase REQUEST_TIMEOUT."
        )
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


if "example_to_process" in st.session_state:
    prompt = st.session_state.example_to_process
    del st.session_state.example_to_process
    
    if not groq_api_key:
        st.error("Please add your Groq API key in the sidebar to continue.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            result = process_query(prompt, groq_api_key, selected_model)
        
        if result:
            # Display final answer
            final_answer = result.get("final_answer", "No response generated.")
            st.markdown(final_answer)
            
            # Store message with info
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "generated_code": result.get("generated_code", ""),
                "code_execution_error": result.get("code_execution_error", ""),
                "code_execution_iterations": result.get("code_execution_iterations", 1),
                "selected_experts": result.get("selected_experts", []),
                "expert_responses": result.get("expert_responses", {}),
                "execution_plan": result.get("execution_plan", {}),
                "token_usage": result.get("token_usage", {}),
            })
            
            with st.expander("🔬 Orchestration Insights", expanded=True):
                render_result_panels(result, key_prefix="example_current")
    
    if result:
        st.rerun()

# Regular chat input
if prompt := st.chat_input(placeholder="Ask me anything..."):
    if not groq_api_key:
        st.error("Please add your Groq API key in the sidebar to continue.")
        st.stop()
    
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Process and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Processing your query..."):
            result = process_query(prompt, groq_api_key, selected_model)
        
        if result:
            # Display final answer
            final_answer = result.get("final_answer", "No response generated.")
            st.markdown(final_answer)
            
            # Store message with info
            st.session_state.messages.append({
                "role": "assistant",
                "content": final_answer,
                "generated_code": result.get("generated_code", ""),
                "code_execution_error": result.get("code_execution_error", ""),
                "code_execution_iterations": result.get("code_execution_iterations", 1),
                "selected_experts": result.get("selected_experts", []),
                "expert_responses": result.get("expert_responses", {}),
                "execution_plan": result.get("execution_plan", {}),
                "token_usage": result.get("token_usage", {}),
            })
            
            with st.expander("🔬 Orchestration Insights", expanded=True):
                render_result_panels(result, key_prefix="chat_current")

    if result:
        st.rerun()

