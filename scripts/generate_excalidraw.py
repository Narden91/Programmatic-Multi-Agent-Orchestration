"""Generate a clean, light-mode Excalidraw diagram for MOSAIC-MoE Architecture."""

import json
from pathlib import Path


def create_element(
    el_type: str,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str = "",
    stroke_color: str = "#0969da",
    bg_color: str = "#ffffff",
    fill_style: str = "solid",
    stroke_width: int = 2,
    font_size: int = 16,
    text_align: str = "center",
    roundness: dict = None,
    points: list = None,
    el_id: str = "",
):
    el = {
        "id": el_id or f"el_{x}_{y}_{el_type}",
        "type": el_type,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "angle": 0,
        "strokeColor": stroke_color,
        "backgroundColor": bg_color,
        "fillStyle": fill_style,
        "strokeWidth": stroke_width,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "roundness": roundness or {"type": 3},
        "seed": 1000 + int(abs(x) + abs(y)) % 9000,
        "version": 1,
        "versionNonce": 1,
        "isDeleted": False,
        "boundElements": None,
        "updated": 1,
        "link": None,
        "locked": False,
    }
    if el_type == "text":
        el.update({
            "text": text,
            "fontSize": font_size,
            "fontFamily": 1,
            "textAlign": text_align,
            "verticalAlign": "middle",
            "baseline": font_size,
            "containerId": None,
            "originalText": text,
            "lineHeight": 1.25,
        })
    elif el_type == "arrow" and points:
        el.update({
            "points": points,
            "lastCommittedPoint": None,
            "startBinding": None,
            "endBinding": None,
            "startArrowhead": None,
            "endArrowhead": "arrow",
        })
    return el


def build_mosaic_excalidraw_light() -> dict:
    elements = []

    # Canvas Title (Dark Blue / Slate)
    elements.append(create_element(
        "text", 360, 40, 680, 40,
        text="MOSAIC-MoE v5/v6: Programmatic Multi-Agent Architecture",
        stroke_color="#0969da", bg_color="transparent", font_size=24, el_id="title_main"
    ))
    elements.append(create_element(
        "text", 380, 80, 640, 30,
        text="Code-as-Orchestration • AST Pre-Flight Sandbox • Semantic Atom Memory • Motif Compression",
        stroke_color="#57606a", bg_color="transparent", font_size=14, el_id="title_sub"
    ))

    # 1. User Input Box (Light Blue)
    elements.append(create_element(
        "rectangle", 60, 160, 200, 100,
        stroke_color="#0969da", bg_color="#ddf4ff", roundness={"type": 3}, el_id="box_user"
    ))
    elements.append(create_element(
        "text", 70, 185, 180, 50,
        text="1. User Task / Query\n(Natural Language)",
        stroke_color="#0969da", bg_color="transparent", font_size=15, el_id="txt_user"
    ))

    # Arrow 1: User -> Memory Retrieval
    elements.append(create_element(
        "arrow", 260, 210, 80, 0,
        points=[[0, 0], [80, 0]], stroke_color="#0969da", el_id="arr_u_to_mem"
    ))

    # 2. Semantic Memory Graph Box (Light Purple)
    elements.append(create_element(
        "rectangle", 340, 140, 230, 140,
        stroke_color="#8250df", bg_color="#fbefff", roundness={"type": 3}, el_id="box_mem"
    ))
    elements.append(create_element(
        "text", 350, 155, 210, 110,
        text="2. Semantic Memory Graph\n• Atom Edges Topology\n• Discovered Motifs\n• Few-Shot Plan Retrieval\n(SQLite + MiniLM)",
        stroke_color="#3b2300", bg_color="transparent", font_size=13, text_align="left", el_id="txt_mem"
    ))

    # Arrow 2: Memory -> Orchestrator
    elements.append(create_element(
        "arrow", 570, 210, 80, 0,
        points=[[0, 0], [80, 0]], stroke_color="#8250df", el_id="arr_mem_to_orch"
    ))

    # 3. Orchestrator Synthesizer Box (Light Blue)
    elements.append(create_element(
        "rectangle", 650, 140, 240, 140,
        stroke_color="#0969da", bg_color="#ddf4ff", roundness={"type": 3}, el_id="box_orch"
    ))
    elements.append(create_element(
        "text", 660, 155, 220, 110,
        text="3. Orchestrator Synthesizer\n(Frontier LLM / Groq LPU)\n• Synthesizes Python DAG\n• async def orchestrate():\n• query_agent(expert, q)",
        stroke_color="#1f2328", bg_color="transparent", font_size=13, text_align="left", el_id="txt_orch"
    ))

    # Arrow 3: Orchestrator -> AST Verifier
    elements.append(create_element(
        "arrow", 890, 210, 80, 0,
        points=[[0, 0], [80, 0]], stroke_color="#0969da", el_id="arr_orch_to_ast"
    ))

    # 4. AST Pre-Flight Verifier (Light Amber/Yellow)
    elements.append(create_element(
        "rectangle", 970, 140, 220, 140,
        stroke_color="#bf8700", bg_color="#fff8c5", roundness={"type": 3}, el_id="box_ast"
    ))
    elements.append(create_element(
        "text", 980, 155, 200, 110,
        text="4. AST Pre-Flight Verifier\n• Static Syntax Inspection\n• Whitelist Safe Builtins\n• Block eval/exec/imports\n• Enforce unawaited await",
        stroke_color="#4d2d00", bg_color="transparent", font_size=13, text_align="left", el_id="txt_ast"
    ))

    # Arrow 4: AST -> Isolated Sandbox Runtime (Downward Green)
    elements.append(create_element(
        "arrow", 1080, 280, 0, 70,
        points=[[0, 0], [0, 70]], stroke_color="#1a7f37", el_id="arr_ast_to_sb"
    ))

    # 5. Sandboxed DAG Execution Engine (Light Green Outer Container)
    elements.append(create_element(
        "rectangle", 260, 350, 930, 220,
        stroke_color="#1a7f37", bg_color="#dafbe1", roundness={"type": 3}, stroke_width=2, el_id="box_sb_outer"
    ))
    elements.append(create_element(
        "text", 280, 365, 890, 30,
        text="5. Isolated Sandboxed Async DAG Runtime (asyncio.gather / concurrent tasks)",
        stroke_color="#116329", bg_color="transparent", font_size=15, text_align="left", el_id="txt_sb_head"
    ))

    # Expert Micro-Agents inside Sandbox (White Cards with Slate Borders)
    experts = [
        ("Technical Expert", "Python, Math, Specs", 290, 420),
        ("Analytical Expert", "Critique, Logic, Schema", 520, 420),
        ("Creative Expert", "Analogy, Story, Voice", 750, 420),
        ("Domain Expert", "Legal, Research, Custom", 980, 420),
    ]
    for name, desc, ex, ey in experts:
        elements.append(create_element(
            "rectangle", ex, ey, 190, 90,
            stroke_color="#0969da", bg_color="#ffffff", roundness={"type": 3}, el_id=f"box_exp_{name[:4]}"
        ))
        elements.append(create_element(
            "text", ex + 10, ey + 15, 170, 60,
            text=f"🤖 {name}\n({desc})",
            stroke_color="#1f2328", bg_color="transparent", font_size=12, el_id=f"txt_exp_{name[:4]}"
        ))

    # Arrow 5: Sandbox -> Final Aggregator (Downward Blue)
    elements.append(create_element(
        "arrow", 725, 570, 0, 60,
        points=[[0, 0], [0, 60]], stroke_color="#0969da", el_id="arr_sb_to_agg"
    ))

    # 6. Aggregated Output & Claim Extractor Box (Light Blue)
    elements.append(create_element(
        "rectangle", 580, 630, 290, 110,
        stroke_color="#0969da", bg_color="#ddf4ff", roundness={"type": 3}, el_id="box_agg"
    ))
    elements.append(create_element(
        "text", 590, 645, 270, 80,
        text="6. Output Resolution & Claims\n• Verified Final Answer\n• Token & Latency Telemetry\n• Semantic Atom Extraction",
        stroke_color="#1f2328", bg_color="transparent", font_size=13, text_align="left", el_id="txt_agg"
    ))

    # Arrow 6: Feedback Loop from Aggregator back to Memory Graph with Compression
    elements.append(create_element(
        "arrow", 580, 685, -450, -430,
        points=[[0, 0], [-200, 0], [-200, -450], [-130, -450]],
        stroke_color="#8250df", el_id="arr_loop_compression"
    ))
    elements.append(create_element(
        "rectangle", 120, 480, 210, 90,
        stroke_color="#8250df", bg_color="#fbefff", roundness={"type": 3}, el_id="box_comp_badge"
    ))
    elements.append(create_element(
        "text", 125, 490, 200, 70,
        text="7. Online Distillation\n• Motif Dictionary Coder\n• -54.1% Footprint (Zlib)\n• Persist atom_edges",
        stroke_color="#3b2300", bg_color="transparent", font_size=12, text_align="left", el_id="txt_comp_badge"
    ))

    return {
        "type": "excalidraw",
        "version": 2,
        "source": "https://excalidraw.com",
        "elements": elements,
        "appState": {
            "viewBackgroundColor": "#ffffff",
            "currentItemFontFamily": 1,
            "gridSize": None,
        },
        "files": {},
    }


def main():
    excal_data = build_mosaic_excalidraw_light()
    out_path = Path("artifacts/mosaic_architecture.excalidraw")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(excal_data, indent=2), encoding="utf-8")
    print(f"Wrote light-theme Excalidraw architecture diagram to {out_path}")


if __name__ == "__main__":
    main()
