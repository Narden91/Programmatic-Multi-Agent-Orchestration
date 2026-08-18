"""Generate an updated light-theme Excalidraw diagram for MOSAIC-MoE v5/v6 Architecture."""

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
        "text", 360, 30, 720, 35,
        text="MOSAIC-MoE v5/v6: Complete Programmatic Architecture",
        stroke_color="#0969da", bg_color="transparent", font_size=24, el_id="title_main"
    ))
    elements.append(create_element(
        "text", 370, 70, 700, 25,
        text="Code-as-Orchestration • Caveman/Ponytail Skills • AST Pre-Flight Sandbox • Motif Compression",
        stroke_color="#57606a", bg_color="transparent", font_size=13, el_id="title_sub"
    ))

    # 1. User Input Box (Light Blue)
    elements.append(create_element(
        "rectangle", 50, 140, 200, 110,
        stroke_color="#0969da", bg_color="#ddf4ff", roundness={"type": 3}, el_id="box_user"
    ))
    elements.append(create_element(
        "text", 60, 160, 180, 70,
        text="1. User Task / Query\n(Natural Language)\n(FastAPI / React UI)",
        stroke_color="#0969da", bg_color="transparent", font_size=14, el_id="txt_user"
    ))

    # Arrow 1: User -> Memory Retrieval
    elements.append(create_element(
        "arrow", 250, 195, 80, 0,
        points=[[0, 0], [80, 0]], stroke_color="#0969da", el_id="arr_u_to_mem"
    ))

    # 2. Semantic Memory Graph Box (Light Purple)
    elements.append(create_element(
        "rectangle", 330, 130, 240, 140,
        stroke_color="#8250df", bg_color="#fbefff", roundness={"type": 3}, el_id="box_mem"
    ))
    elements.append(create_element(
        "text", 340, 145, 220, 110,
        text="2. Semantic Memory Graph\n• Subgraph Atom-Edges\n• Discovered Plan Motifs\n• Top-k Graph Neighborhood\n(SQLite + MiniLM-L6-v2)",
        stroke_color="#3b2300", bg_color="transparent", font_size=12.5, text_align="left", el_id="txt_mem"
    ))

    # Arrow 2: Memory -> Orchestrator
    elements.append(create_element(
        "arrow", 570, 195, 80, 0,
        points=[[0, 0], [80, 0]], stroke_color="#8250df", el_id="arr_mem_to_orch"
    ))

    # 3. Orchestrator Synthesizer Box (Light Blue)
    elements.append(create_element(
        "rectangle", 650, 120, 270, 160,
        stroke_color="#0969da", bg_color="#ddf4ff", roundness={"type": 3}, el_id="box_orch"
    ))
    elements.append(create_element(
        "text", 660, 135, 250, 130,
        text="3. Orchestrator Synthesizer\n(Groq LPU / GPT-OSS / Claude 5)\n• Karpathy DAG Principles\n• Single-Pass Python Generation\n• Assigns Caveman / Ponytail\n• query_agent(..., weight, skill)",
        stroke_color="#1f2328", bg_color="transparent", font_size=12.5, text_align="left", el_id="txt_orch"
    ))

    # Arrow 3: Orchestrator -> AST Verifier
    elements.append(create_element(
        "arrow", 920, 195, 70, 0,
        points=[[0, 0], [70, 0]], stroke_color="#0969da", el_id="arr_orch_to_ast"
    ))

    # 4. AST Pre-Flight Verifier (Light Amber/Yellow)
    elements.append(create_element(
        "rectangle", 990, 120, 240, 160,
        stroke_color="#bf8700", bg_color="#fff8c5", roundness={"type": 3}, el_id="box_ast"
    ))
    elements.append(create_element(
        "text", 1000, 135, 220, 130,
        text="4. AST Pre-Flight Sandbox Guard\n• Speculative asyncio.gather\n• Safe Builtins Whitelist\n• Block eval/exec/open/imports\n• Rejects Unawaited Coroutines\n• 100% Pass Rate (170/170)",
        stroke_color="#4d2d00", bg_color="transparent", font_size=12, text_align="left", el_id="txt_ast"
    ))

    # Arrow 4: AST -> Isolated Sandbox Runtime (Downward Green)
    elements.append(create_element(
        "arrow", 1110, 280, 0, 70,
        points=[[0, 0], [0, 70]], stroke_color="#1a7f37", el_id="arr_ast_to_sb"
    ))

    # 5. Sandboxed DAG Execution Engine (Light Green Outer Container)
    elements.append(create_element(
        "rectangle", 240, 350, 990, 230,
        stroke_color="#1a7f37", bg_color="#dafbe1", roundness={"type": 3}, stroke_width=2, el_id="box_sb_outer"
    ))
    elements.append(create_element(
        "text", 260, 365, 950, 25,
        text="5. Isolated Sandboxed Async DAG Runtime with Dynamic Agent Skills (asyncio.gather)",
        stroke_color="#116329", bg_color="transparent", font_size=14.5, text_align="left", el_id="txt_sb_head"
    ))

    # Expert Micro-Agents inside Sandbox (White Cards with Skills)
    experts = [
        ("Technical Expert", "PONYTAIL MODE (weight=0.9)\nHigh Rigor & Code Invariants", 270, 410, "#0969da", "#eff6ff"),
        ("Analytical Expert", "PONYTAIL MODE (weight=0.8)\nFirst-Principles Logic & Proofs", 510, 410, "#0969da", "#eff6ff"),
        ("Auxiliary Validator", "CAVEMAN MODE (weight=0.2)\nDense Telegrams (-60% tokens)", 750, 410, "#d97706", "#fffbeb"),
        ("Domain Expert", "CAVEMAN MODE (weight=0.3)\nRaw Bullet Facts / No Fluff", 990, 410, "#d97706", "#fffbeb"),
    ]
    for name, desc, ex, ey, border_c, bg_c in experts:
        elements.append(create_element(
            "rectangle", ex, ey, 210, 110,
            stroke_color=border_c, bg_color=bg_c, roundness={"type": 3}, el_id=f"box_exp_{name[:4]}"
        ))
        elements.append(create_element(
            "text", ex + 10, ey + 15, 190, 80,
            text=f"🤖 {name}\n\n{desc}",
            stroke_color="#1f2328", bg_color="transparent", font_size=11.5, text_align="center", el_id=f"txt_exp_{name[:4]}"
        ))

    # Arrow 5: Sandbox -> Final Aggregator (Downward Blue)
    elements.append(create_element(
        "arrow", 735, 580, 0, 60,
        points=[[0, 0], [0, 60]], stroke_color="#0969da", el_id="arr_sb_to_agg"
    ))

    # 6. Aggregated Output & Natural Language Translator (Light Blue)
    elements.append(create_element(
        "rectangle", 560, 640, 350, 120,
        stroke_color="#0969da", bg_color="#ddf4ff", roundness={"type": 3}, el_id="box_agg"
    ))
    elements.append(create_element(
        "text", 570, 655, 330, 90,
        text="6. Natural Language Translation & Output\n• Synthesizes Caveman notes into fluent prose\n• Emits verified claims & semantic atoms\n• Records Latency & Token Telemetry\n• Formats clean user deliverable",
        stroke_color="#1f2328", bg_color="transparent", font_size=12.5, text_align="left", el_id="txt_agg"
    ))

    # Arrow 6: Feedback Loop from Aggregator back to Memory Graph with Compression
    elements.append(create_element(
        "arrow", 560, 700, -470, -460,
        points=[[0, 0], [-230, 0], [-230, -480], [-100, -480]],
        stroke_color="#8250df", el_id="arr_loop_compression"
    ))
    elements.append(create_element(
        "rectangle", 70, 480, 220, 110,
        stroke_color="#8250df", bg_color="#fbefff", roundness={"type": 3}, el_id="box_comp_badge"
    ))
    elements.append(create_element(
        "text", 75, 495, 210, 80,
        text="7. Online Distillation Loop\n• MotifDictionaryCoder\n• -54.1% Storage Footprint\n• Persists atom_edges to SQLite\n• Decompression <0.09 ms",
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
    print(f"Wrote updated light-theme Excalidraw architecture diagram to {out_path}")


if __name__ == "__main__":
    main()
