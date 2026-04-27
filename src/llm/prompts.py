from typing import Any, Dict, List, Optional, Tuple


class OrchestratorPrompts:
    """Prompts for the Code Orchestrator agent"""
    
    @staticmethod
    def create_orchestration_prompt(
        query: str,
        available_experts: List[str],
        expert_descriptions: Optional[Dict[str, str]] = None,
        few_shot_examples: Optional[List[Tuple[str, str]]] = None,
        atom_few_shot_examples: Optional[List[Dict[str, Any]]] = None,
        atom_graph_examples: Optional[List[Dict[str, Any]]] = None,
        plan_graph_examples: Optional[List[Dict[str, Any]]] = None,
        conversation_context: str = "",
    ) -> str:
        """Create orchestration prompt for code generation.

        Parameters
        ----------
        query : str
            The user's question.
        available_experts : list[str]
            Expert types the sandbox can call.
        expert_descriptions : dict[str, str] | None
            ``{expert_type: description}`` — when provided, used instead of
            the built-in fallback descriptions.
        few_shot_examples : list[tuple[str, str]] | None
            ``[(query, code), ...]`` past successful scripts to include as
            few-shot examples.
        conversation_context : str
            Formatted multi-turn history to include in the prompt.
        """
        # Fall back to built-in descriptions when the registry isn't passed
        _fallback = {
            "technical": "programming, technology, mathematics, sciences",
            "creative": "brainstorming, storytelling, creative content",
            "analytical": "data analysis, comparisons, logical decisions",
            "general": "general conversation, facts, basic information",
        }
        descs = expert_descriptions or _fallback

        experts_list = "\n".join([
            f"  * '{expert}' : for {descs.get(expert, 'general queries')}"
            for expert in available_experts
        ])

        # Optional few-shot section
        few_shot_section = ""
        if few_shot_examples:
            examples = []
            for i, (ex_query, ex_code) in enumerate(few_shot_examples, 1):
                examples.append(
                    f"--- Example {i} ---\n"
                    f'Query: "{ex_query}"\n'
                    f"```python\n{ex_code}\n```"
                )
            few_shot_section = (
                "\n\nHere are examples of previously successful orchestration scripts:\n"
                + "\n\n".join(examples)
                + "\n\nUse these as inspiration, but adapt to the current query.\n"
            )

        atom_few_shot_section = ""
        if atom_few_shot_examples:
            examples = []
            for i, example in enumerate(atom_few_shot_examples, 1):
                dependencies = ", ".join(example.get("dependencies") or []) or "none"
                evidence_tags = ", ".join(example.get("evidence_tags") or []) or "none"
                examples.append(
                    f"--- Atom Hint {i} ---\n"
                    f'Origin Query: "{example.get("task_description", "")}"\n'
                    f"Expert: {example.get('agent_type', 'general')}\n"
                    f"Atom: {example.get('text', '')}\n"
                    f"Confidence: {float(example.get('confidence', 0.0)):.2f}\n"
                    f"Dependencies: {dependencies}\n"
                    f"Evidence Tags: {evidence_tags}"
                )
            atom_few_shot_section = (
                "\n\nRelevant semantic atoms retrieved from prior successful orchestrations:\n"
                + "\n\n".join(examples)
                + "\n\nUse these atoms as high-signal hints for decomposition, expert choice, and dependency ordering. Do not copy them blindly.\n"
            )

        atom_graph_section = ""
        if atom_graph_examples:
            examples = []
            for i, example in enumerate(atom_graph_examples, 1):
                neighbors = example.get("neighbors") or []
                if neighbors:
                    neighbor_text = "; ".join(
                        f"{neighbor.get('atom_id', '')}: {neighbor.get('text', '')}"
                        for neighbor in neighbors
                    )
                else:
                    neighbor_text = "none"

                edges = example.get("edges") or []
                if edges:
                    edge_text = "; ".join(
                        f"{edge.get('source_atom_id', '')} -> {edge.get('target_atom_id', '')} ({edge.get('edge_type', 'dependency')})"
                        for edge in edges
                    )
                else:
                    edge_text = "none"

                examples.append(
                    f"--- Atom Graph Hint {i} ---\n"
                    f'Origin Query: "{example.get("task_description", "")}"\n'
                    f"Expert: {example.get('agent_type', 'general')}\n"
                    f"Seed Atom: {example.get('seed_atom_id', '')}: {example.get('seed_text', '')}\n"
                    f"Similarity: {float(example.get('similarity', 0.0)):.2f}\n"
                    f"Neighbor Atoms: {neighbor_text}\n"
                    f"Dependency Edges: {edge_text}"
                )
            atom_graph_section = (
                "\n\nRelevant atom graph neighborhoods from prior successful orchestrations:\n"
                + "\n\n".join(examples)
                + "\n\nUse these graph neighborhoods to preserve dependency order, decide which expert outputs should precede others, and recognize reusable plan motifs.\n"
            )

        plan_graph_section = ""
        if plan_graph_examples:
            examples = []
            for i, example in enumerate(plan_graph_examples, 1):
                parallel_label = "parallel" if example.get("is_parallel") else "sequential"
                group_text = f", group {example.get('group_id')}" if example.get("group_id") is not None else ""
                examples.append(
                    f"--- Plan Motif {i} ---\n"
                    f'Origin Query: "{example.get("task_description", "")}"\n'
                    f"Motif: {example.get('motif_text', '')}\n"
                    f"Expert: {example.get('expert_type', 'unknown')}\n"
                    f"Execution: {parallel_label}{group_text}\n"
                    f"Similarity: {float(example.get('similarity', 0.0)):.2f}"
                )
            plan_graph_section = (
                "\n\nRelevant compressed plan motifs from prior successful orchestrations:\n"
                + "\n\n".join(examples)
                + "\n\nUse these motifs as reusable scheduling patterns when deciding whether to stay sequential or parallel.\n"
            )
        
        # Optional conversation context section
        context_section = ""
        if conversation_context:
            context_section = (
                f"\n\n{conversation_context}\n\n"
                "Take the conversation history above into account when "
                "answering the current query.\n"
            )

        return f"""You are an advanced AI orchestrator. Your task is to write an async Python script that solves the user's query by programmatically calling expert micro-agents.

User Query: "{query}"
{context_section}
You have access to the following async functions (tools) in your sandbox environment:
- query_agent(agent_type: str, prompt: str) -> AgentResult : queries an expert. Default to using `.text` on the result (e.g. `res = await query_agent('technical', '...'); print(res.text)`). When helpful, you may also inspect `res.atoms`, a list of semantic atoms with fields like `atom_id`, `text`, `confidence`, `dependencies`, and `evidence_tags`. Available agent_types:
{experts_list}
- memory_store(key: str, text: str, metadata: dict = None) -> str : stores text in ephemeral vector database
- memory_search(query: str, top_k: int = 5) -> list[dict] : retrieves top_k semantically relevant chunks
- compress_context(query: str, top_k: int = 5) -> str : returns a summarized string of relevant memory chunks
- asyncio.gather(*tasks) # for running agents in parallel

Instructions:
1. Break down the query into steps.
2. For each step, determine if you need to query an expert. If you have multiple independent questions or chunks, use a `for` loop to build a list of tasks and `await asyncio.gather(*tasks)`.
3. If the flow is sequential, await each expert one by one and pass previous responses if necessary.
3a. Never access `.text`, `.atoms`, or other result fields on `query_agent(...)` until after you have awaited that call or received the resolved result from `asyncio.gather`.
4. You MUST define exactly ONE function called `async def orchestrate():` with no parameters.
5. In your `orchestrate` function, write the logic to gather the information, and then clearly RETURN a single final string with the comprehensive answer based on everything gathered. Do NOT print the final answer instead of returning it.
5a. The returned string must be the user-facing deliverable, not meta-commentary about how good the draft is. If the user asks for a story, return the story itself. If the user asks for code, return the code or explanation itself. Keep evaluation, critique, scoring, and improvement notes internal unless the user explicitly asked for critique.
5b. Do NOT return editorial phrases like "Based on the analysis", "To further refine", "areas for improvement", or quality-review summaries unless the user explicitly requested that kind of analysis.
6. Only output valid Python code inside a single ```python codeblock. Do not output anything outside the codeblock.
7. You may use standard Python string manipulation and conditional logic (`if`/`else`).
8. You do NOT need to import the tool functions or `asyncio`; they are already injected into the global namespace.
{few_shot_section}
{atom_few_shot_section}
{atom_graph_section}
{plan_graph_section}
Example:
```python
async def orchestrate():
    technical_task = query_agent("technical", "Explain the architecture part of: " + user_query_fragment)
    analytical_task = query_agent("analytical", "Analyze the data part of: " + user_query_fragment)
    
    tech_result, analytical_result = await asyncio.gather(technical_task, analytical_task)
    
    final_synthesis = await query_agent("general", f"Combine these: {{tech_result.text}} and {{analytical_result.text}} into a cohesive summary.")
    return final_synthesis.text
```

Write the orchestration code for the User Query now:"""

    @staticmethod
    def create_retry_prompt(
        query: str,
        failed_code: str,
        error: str,
        available_experts: List[str],
        expert_descriptions: Optional[Dict[str, str]] = None,
    ) -> str:
        """Create a retry prompt that includes the failing code and error."""
        _fallback = {
            "technical": "programming, technology, mathematics, sciences",
            "creative": "brainstorming, storytelling, creative content",
            "analytical": "data analysis, comparisons, logical decisions",
            "general": "general conversation, facts, basic information",
        }
        descs = expert_descriptions or _fallback

        experts_list = "\n".join([
            f"  * '{expert}' : for {descs.get(expert, 'general queries')}"
            for expert in available_experts
        ])

        return f"""You are an advanced AI orchestrator. A previously generated script FAILED during execution. Your job is to fix the script.

User Query: "{query}"

Available async functions (tools):
- query_agent(agent_type: str, prompt: str) -> AgentResult : queries an expert. Use `.text` by default on the returned AgentResult object. When it helps, you may inspect `.atoms` for structured semantic claims. Available agent_types:
{experts_list}
- memory_store(key: str, text: str, metadata: dict = None) -> str
- memory_search(query: str, top_k: int = 5) -> list[dict]
- compress_context(query: str, top_k: int = 5) -> str
- asyncio.gather(*tasks)

--- FAILED SCRIPT ---
```python
{failed_code}
```

--- ERROR ---
{error}

--- INSTRUCTIONS ---
1. Analyze the error and the failed script above.
2. Fix the issue and rewrite the script.
2a. If the failure mentions `.text`, `.atoms`, or a tracked query handle, await the `query_agent(...)` call before accessing result fields.
3. You MUST define exactly ONE function called `async def orchestrate():` with no parameters.
4. The function must RETURN a single final string.
4a. That returned string must be the direct answer to the user's request, not critique, evaluation notes, or suggestions for future refinement unless the user explicitly asked for critique.
5. Only output valid Python code inside a single ```python codeblock.
6. You do NOT need to import the tool functions or `asyncio`; they are already in the global namespace.
7. Do NOT use `import` statements — they are blocked by the sandbox.

Write the fixed orchestration code now:"""