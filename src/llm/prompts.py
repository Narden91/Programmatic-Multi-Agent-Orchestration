from typing import Dict, List


class RouterPrompts:
    """Prompts for the router agent"""
    
    @staticmethod
    def create_routing_prompt(query: str, available_experts: List[str]) -> str:
        """Create routing prompt"""
        experts_desc = {
            "technical": "programming, technology, mathematics, sciences",
            "creative": "brainstorming, storytelling, creative content",
            "analytical": "data analysis, comparisons, logical decisions",
            "general": "general conversation, facts, basic information"
        }
        
        experts_list = "\n".join([
            f"- {expert}: for {experts_desc.get(expert, 'general queries')}"
            for expert in available_experts
        ])
        
        return f"""You are an intelligent router. Analyze this query and select the most appropriate expert(s).

Query: "{query}"

Available experts:
{experts_list}

Instructions:
- Select ONE or MORE experts (up to 3 for complex queries)
- Respond ONLY with expert names separated by commas
- Examples: "technical", "creative,analytical", "technical,creative,analytical"

Selected experts:"""


class ExpertPrompts:
    """Prompts for expert agents"""
    
    @staticmethod
    def create_technical_prompt(query: str) -> str:
        """Create technical expert prompt"""
        return f"""You are a technical expert specialized in programming, technology, and sciences.

Query: "{query}"

Provide a response that is:
- Precise and detailed
- Includes concrete examples if relevant
- Uses appropriate technical terminology
- Follows best practices

Response:"""
    
    @staticmethod
    def create_creative_prompt(query: str) -> str:
        """Create creative expert prompt"""
        return f"""You are a creative expert specialized in storytelling, brainstorming, and original content.

Query: "{query}"

Provide a response that is:
- Innovative and original
- Engaging and interesting
- Uses creative metaphors or analogies
- Thinks outside the box

Response:"""
    
    @staticmethod
    def create_analytical_prompt(query: str) -> str:
        """Create analytical expert prompt"""
        return f"""You are an analytical expert specialized in data analysis, logic, and rational decisions.

Query: "{query}"

Provide a response that is:
- Structured and methodical
- Includes pros/cons if applicable
- Based on data and facts
- Uses step-by-step reasoning

Response:"""
    
    @staticmethod
    def create_general_prompt(query: str) -> str:
        """Create general expert prompt"""
        return f"""You are a general knowledge expert, friendly and conversational.

Query: "{query}"

Provide a response that is:
- Clear and understandable
- Friendly and conversational
- Complete but concise
- Suitable for all audiences

Response:"""


class SynthesizerPrompts:
    """Prompts for the synthesizer agent"""
    
    @staticmethod
    def create_synthesis_prompt(query: str, expert_responses: Dict[str, str]) -> str:
        """Create synthesis prompt"""
        responses_text = "\n\n".join([
            f"=== {expert.upper()} EXPERT ===\n{response}"
            for expert, response in expert_responses.items()
        ])
        
        return f"""You are an expert synthesizer. You have received responses from multiple specialized experts.

Original Query: "{query}"

Expert Responses:
{responses_text}

Your task:
1. Identify key points from each expert
2. Combine insights into a coherent response
3. Resolve any contradictions
4. Create a final answer that is better than the sum of its parts

Synthesized final response:"""


class OrchestratorPrompts:
    """Prompts for the Code Orchestrator agent"""
    
    @staticmethod
    def create_orchestration_prompt(query: str, available_experts: List[str]) -> str:
        """Create orchestration prompt for code generation"""
        experts_desc = {
            "technical": "programming, technology, mathematics, sciences",
            "creative": "brainstorming, storytelling, creative content",
            "analytical": "data analysis, comparisons, logical decisions",
            "general": "general conversation, facts, basic information"
        }
        
        experts_list = "\n".join([
            f"- query_{expert}_expert(prompt: str) -> str : for {experts_desc.get(expert, 'general queries')}"
            for expert in available_experts
        ])
        
        return f"""You are an advanced AI orchestrator. Your task is to write an async Python script that solves the user's query by programmatically calling expert micro-agents.

User Query: "{query}"

You have access to the following async functions (tools) in your sandbox environment:
{experts_list}
- asyncio.gather(*tasks) # for running agents in parallel

Instructions:
1. Break down the query into steps.
2. For each step, determine if you need to query an expert. If you have multiple independent questions or chunks, use a `for` loop to build a list of tasks and `await asyncio.gather(*tasks)`.
3. If the flow is sequential, await each expert one by one and pass previous responses if necessary.
4. You MUST define exactly ONE function called `async def orchestrate():` with no parameters.
5. In your `orchestrate` function, write the logic to gather the information, and then clearly RETURN a single final string with the comprehensive answer based on everything gathered. Do NOT print the final answer instead of returning it.
6. Only output valid Python code inside a single ```python codeblock. Do not output anything outside the codeblock.
7. You may use standard Python string manipulation and conditional logic (`if`/`else`).
8. You do NOT need to import the tool functions or `asyncio`; they are already injected into the global namespace.

Example:
```python
async def orchestrate():
    technical_task = query_technical_expert("Explain the architecture part of: " + user_query_fragment)
    analytical_task = query_analytical_expert("Analyze the data part of: " + user_query_fragment)
    
    tech_response, analytical_response = await asyncio.gather(technical_task, analytical_task)
    
    final_synthesis = await query_general_expert(f"Combine these: {{tech_response}} and {{analytical_response}} into a cohesive summary.")
    return final_synthesis
```

Write the orchestration code for the User Query now:"""