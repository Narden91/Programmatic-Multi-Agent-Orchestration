import plotly.graph_objects as go
from typing import Dict, List, Optional

from src.utils.code_analyzer import analyze_code, ExecutionPlan


class MoEVisualizer:
    """Visualizations for the programmatic MoE orchestration system.

    The graph reflects the actual architecture:
        Query → Orchestrator → Sandbox → [Expert-1 … Expert-N] → Answer
    """

    EXPERT_COLORS = {
        "Query": "#a8b5ff",
        "Orchestrator": "#c9b3ff",
        "Sandbox": "#fcd34d",
        "technical": "#fca5a5",
        "creative": "#86efac",
        "analytical": "#93c5fd",
        "general": "#d8b4fe",
        "Answer": "#fdba74",
    }

    # ------------------------------------------------------------------
    # Static architecture graph
    # ------------------------------------------------------------------

    @classmethod
    def create_network_graph(
        cls,
        selected_experts: List[str],
        expert_responses: Dict[str, str],
        generated_code: Optional[str] = None,
    ) -> go.Figure:
        """Create network graph showing the programmatic orchestration flow."""
        nodes = ["Query", "Orchestrator", "Sandbox"] + selected_experts + ["Answer"]
        pos = cls._calculate_node_positions(nodes, selected_experts)
        edge_x, edge_y = cls._create_edges(pos, selected_experts)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=3, color="#e5e7eb"),
            hoverinfo="none",
            mode="lines",
        )

        node_trace = cls._create_node_trace(nodes, pos)

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=350,
                plot_bgcolor="#fafafa",
                paper_bgcolor="#ffffff",
            ),
        )

        return fig

    # ------------------------------------------------------------------
    # Dynamic execution-plan graph (from AST analysis)
    # ------------------------------------------------------------------

    @classmethod
    def create_execution_plan_graph(
        cls,
        code: str,
        actual_experts: Optional[List[str]] = None,
    ) -> go.Figure:
        """Visualise the execution plan extracted from generated code.

        Parallel calls (``asyncio.gather``) are shown side-by-side;
        sequential calls form a chain.
        """
        plan = analyze_code(code)

        # Build node list: Start → [calls...] → End
        nodes: List[str] = ["Start"]
        node_colors: List[str] = ["#a8b5ff"]
        node_x: List[float] = [0.0]
        node_y: List[float] = [0.5]
        edges_from: List[int] = []
        edges_to: List[int] = []

        col = 1.0
        prev_indices: List[int] = [0]  # indices of the previous "layer"

        # Group calls into sequential steps and parallel groups
        steps = cls._group_into_steps(plan)

        for step in steps:
            layer_indices: List[int] = []
            n = len(step)
            for i, call in enumerate(step):
                idx = len(nodes)
                label = call.expert_type
                nodes.append(label)
                node_colors.append(cls.EXPERT_COLORS.get(label, "#d1d5db"))
                node_x.append(col)
                node_y.append((i + 1) / (n + 1) if n > 1 else 0.5)
                layer_indices.append(idx)
                # Connect from every node in the previous layer
                for pi in prev_indices:
                    edges_from.append(pi)
                    edges_to.append(idx)
            prev_indices = layer_indices
            col += 1.0

        # End node
        end_idx = len(nodes)
        nodes.append("End")
        node_colors.append("#fdba74")
        node_x.append(col)
        node_y.append(0.5)
        for pi in prev_indices:
            edges_from.append(pi)
            edges_to.append(end_idx)

        # Build Plotly traces
        edge_x: List[Optional[float]] = []
        edge_y: List[Optional[float]] = []
        for fi, ti in zip(edges_from, edges_to):
            edge_x.extend([node_x[fi], node_x[ti], None])
            edge_y.extend([node_y[fi], node_y[ti], None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color="#94a3b8"),
            hoverinfo="none", mode="lines",
        )
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="middle center",
            textfont=dict(size=10, color="#1e293b", family="Inter, Arial", weight=600),
            marker=dict(size=44, color=node_colors, line=dict(width=2, color="#fff")),
            hoverinfo="text",
        )

        title_parts = []
        if plan.has_parallel:
            title_parts.append(f"{plan.gather_groups} parallel group(s)")
        if plan.has_sequential:
            title_parts.append("sequential calls")
        subtitle = " + ".join(title_parts) if title_parts else "no expert calls detected"

        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(text=f"Execution Plan  ({subtitle})", font=dict(size=13)),
                showlegend=False, hovermode="closest",
                margin=dict(b=20, l=20, r=20, t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=280,
                plot_bgcolor="#fafafa", paper_bgcolor="#ffffff",
            ),
        )
        return fig

    # ------------------------------------------------------------------
    # Token usage summary bar chart
    # ------------------------------------------------------------------

    @classmethod
    def create_token_usage_chart(cls, token_usage: Dict) -> go.Figure:
        """Create a horizontal bar chart of token usage by agent."""
        by_agent = token_usage.get("by_agent", {})
        if not by_agent:
            # Empty placeholder
            return go.Figure(layout=go.Layout(
                title="Token Usage (no data)",
                height=200, margin=dict(b=20, l=20, r=20, t=40),
            ))

        agents = list(by_agent.keys())
        input_tokens = [by_agent[a]["input"] for a in agents]
        output_tokens = [by_agent[a]["output"] for a in agents]

        fig = go.Figure(data=[
            go.Bar(name="Input", y=agents, x=input_tokens, orientation="h",
                   marker_color="#93c5fd"),
            go.Bar(name="Output", y=agents, x=output_tokens, orientation="h",
                   marker_color="#fca5a5"),
        ])
        total = token_usage.get("total_tokens", 0)
        cost = token_usage.get("estimated_cost_usd", 0)
        fig.update_layout(
            barmode="stack",
            title=dict(text=f"Token Usage — {total:,} total ≈ ${cost:.4f}", font=dict(size=13)),
            height=max(180, 40 * len(agents) + 60),
            margin=dict(b=20, l=80, r=20, t=40),
            legend=dict(orientation="h", y=1.15),
        )
        return fig

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _group_into_steps(plan: ExecutionPlan):
        """Group calls into layers: each gather group is one layer,
        sequential calls are individual layers."""
        from src.utils.code_analyzer import ExpertCall

        steps: List[List[ExpertCall]] = []
        seen_groups: set[int] = set()

        for call in plan.calls:
            if call.is_parallel and call.group_id is not None:
                if call.group_id not in seen_groups:
                    seen_groups.add(call.group_id)
                    # Collect all calls in this group
                    group = [c for c in plan.calls if c.group_id == call.group_id]
                    steps.append(group)
            else:
                steps.append([call])
        return steps

    @classmethod
    def _calculate_node_positions(
        cls, nodes: List[str], selected_experts: List[str]
    ) -> Dict:
        """Calculate x,y positions for nodes."""
        pos = {}
        pos["Query"] = (0, 0.5)
        pos["Orchestrator"] = (1, 0.5)
        pos["Sandbox"] = (2, 0.5)

        num_experts = len(selected_experts)
        for i, expert in enumerate(selected_experts):
            y = (i + 1) / (num_experts + 1)
            pos[expert] = (3, y)

        pos["Answer"] = (4, 0.5)
        return pos

    @classmethod
    def _create_edges(cls, pos: Dict, selected_experts: List[str]) -> tuple:
        """Create edge coordinates."""
        edge_x: list = []
        edge_y: list = []

        # Query → Orchestrator → Sandbox
        for a, b in [("Query", "Orchestrator"), ("Orchestrator", "Sandbox")]:
            edge_x.extend([pos[a][0], pos[b][0], None])
            edge_y.extend([pos[a][1], pos[b][1], None])

        # Sandbox → each expert
        for expert in selected_experts:
            edge_x.extend([pos["Sandbox"][0], pos[expert][0], None])
            edge_y.extend([pos["Sandbox"][1], pos[expert][1], None])

        # Each expert → Answer
        for expert in selected_experts:
            edge_x.extend([pos[expert][0], pos["Answer"][0], None])
            edge_y.extend([pos[expert][1], pos["Answer"][1], None])

        return edge_x, edge_y

    @classmethod
    def _create_node_trace(cls, nodes: List[str], pos: Dict) -> go.Scatter:
        """Create node scatter trace."""
        node_x = [pos[node][0] for node in nodes]
        node_y = [pos[node][1] for node in nodes]
        node_color = [cls.EXPERT_COLORS.get(node, "#d1d5db") for node in nodes]

        return go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            text=nodes,
            textposition="middle center",
            textfont=dict(
                size=11, color="#374151", family="Inter, Arial", weight=600
            ),
            marker=dict(
                size=50, color=node_color, line=dict(width=3, color="#ffffff")
            ),
        )
