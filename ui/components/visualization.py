import plotly.graph_objects as go
from typing import Dict, List


class MoEVisualizer:
    """Visualizations for the MoE system"""
    
    EXPERT_COLORS = {
        "Query": "#a8b5ff",
        "Router": "#c9b3ff",
        "technical": "#fca5a5",
        "creative": "#86efac",
        "analytical": "#93c5fd",
        "general": "#d8b4fe",
        "Synthesizer": "#fcd34d",
        "Answer": "#fdba74"
    }
    
    @classmethod
    def create_network_graph(
        cls, 
        selected_experts: List[str], 
        expert_responses: Dict[str, str]
    ) -> go.Figure:
        """Create network graph showing MoE flow"""
        nodes = ["Query", "Router"] + selected_experts + ["Synthesizer", "Answer"]
        pos = cls._calculate_node_positions(nodes, selected_experts)
        edge_x, edge_y = cls._create_edges(pos, selected_experts)
        
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=3, color='#e5e7eb'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = cls._create_node_trace(nodes, pos)
        
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20, l=20, r=20, t=20),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=350,
                plot_bgcolor='#fafafa',
                paper_bgcolor='#ffffff'
            )
        )
        
        return fig
    
    @classmethod
    def _calculate_node_positions(
        cls, 
        nodes: List[str], 
        selected_experts: List[str]
    ) -> Dict:
        """Calculate x,y positions for nodes"""
        pos = {}
        pos["Query"] = (0, 0.5)
        pos["Router"] = (1, 0.5)
        
        num_experts = len(selected_experts)
        for i, expert in enumerate(selected_experts):
            y = (i + 1) / (num_experts + 1)
            pos[expert] = (2, y)
        
        pos["Synthesizer"] = (3, 0.5)
        pos["Answer"] = (4, 0.5)
        
        return pos
    
    @classmethod
    def _create_edges(
        cls, 
        pos: Dict, 
        selected_experts: List[str]
    ) -> tuple:
        """Create edge coordinates"""
        edge_x = []
        edge_y = []
        
        edge_x.extend([pos["Query"][0], pos["Router"][0], None])
        edge_y.extend([pos["Query"][1], pos["Router"][1], None])
        
        for expert in selected_experts:
            edge_x.extend([pos["Router"][0], pos[expert][0], None])
            edge_y.extend([pos["Router"][1], pos[expert][1], None])
        
        for expert in selected_experts:
            edge_x.extend([pos[expert][0], pos["Synthesizer"][0], None])
            edge_y.extend([pos[expert][1], pos["Synthesizer"][1], None])
        
        edge_x.extend([pos["Synthesizer"][0], pos["Answer"][0], None])
        edge_y.extend([pos["Synthesizer"][1], pos["Answer"][1], None])
        
        return edge_x, edge_y
    
    @classmethod
    def _create_node_trace(cls, nodes: List[str], pos: Dict) -> go.Scatter:
        """Create node scatter trace"""
        node_x = [pos[node][0] for node in nodes]
        node_y = [pos[node][1] for node in nodes]
        node_color = [cls.EXPERT_COLORS.get(node, "#d1d5db") for node in nodes]
        
        return go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=nodes,
            textposition="middle center",
            textfont=dict(size=11, color='#374151', family='Inter, Arial', weight=600),
            marker=dict(
                size=50,
                color=node_color,
                line=dict(width=3, color='#ffffff')
            )
        )
    
