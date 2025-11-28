"""Generate a flowchart describing the shelf-life estimation pipeline.

Run:
    python tools/shelf_life_flowchart.py

Outputs Graphviz source and (if Graphviz binaries are available) a PNG diagram
capturing the logic implemented in RuleBasedShelfLife.predict_shelf_life.
"""

from pathlib import Path

try:
    from graphviz import Digraph
except ImportError as exc:  # pragma: no cover - guide users to install dependency
    raise SystemExit(
        "graphviz package not found. Install with `pip install graphviz` "
        "and ensure Graphviz is installed on your system."
    ) from exc


def build_flowchart() -> Digraph:
    dot = Digraph("ShelfLifeFlow", format="png")
    dot.attr(rankdir="TB", fontsize="12", fontname="Helvetica")
    dot.attr("node", style="rounded,filled", fillcolor="#F0F4F8")
    dot.attr("edge", arrowsize="0.8")

    dot.node("start", "Start", shape="oval", fillcolor="#E3F2FD")
    dot.node(
        "inputs",
        "Inputs:\n• Bean type\n• Defect sequence (type, confidence, count)",
    )
    dot.node(
        "base",
        "Base shelf life = lookup(bean type)\n→ convert to base_months",
    )
    dot.node("loop", "Iterate defects", shape="diamond", fillcolor="#FFF3CD")
    dot.node(
        "score",
        "Update totals:\nweight × confidence × count\n+ per-type counts\n+ cumulative confidence",
        fillcolor="#E8F5E9",
    )
    dot.node(
        "metrics",
        "Compute metrics:\n• avg_detection_confidence\n• defect_percentage = clamp(min(score/45,1.5)×100)",
    )
    dot.node(
        "band",
        "Select severity band (mild/moderate/severe)\n& severity_position",
    )
    dot.node(
        "scale",
        "Interpolate severity_scale from band month_edges & month_peak",
    )
    dot.node(
        "months",
        "predicted_months = max(0.1, base_months × severity_scale)",
    )
    dot.node(
        "guard",
        "Insect damage count > 2?",
        shape="diamond",
        fillcolor="#FFF3CD",
    )
    dot.node(
        "cap",
        "Cap predicted_months ≤ base_months × 0.2",
        fillcolor="#FDECEA",
    )
    dot.node(
        "days",
        "predicted_days = max(0, int(predicted_months × 30))",
    )
    dot.node(
        "confidence",
        "Estimate confidence:\ninterpolate band profile\n– detection penalties\n× avg detection confidence\nclamp to 0.2–0.96",
    )
    dot.node(
        "conf_guard",
        "confidence < threshold?",
        shape="diamond",
        fillcolor="#FFF3CD",
    )
    dot.node("uncertain", "Set category = \"Uncertain\"\nRaise confidence slightly", fillcolor="#FDECEA")
    dot.node(
        "category",
        "Assign status & quality grade:\n• mild → Excellent / Grade A\n• moderate → Good or Warning / Grade B or C\n• severe → Critical / Grade D",
    )
    dot.node(
        "output",
        "Output payload:\n• predicted_days, estimated_months, range\n• category, quality_grade, severity & position\n• defect_percentage, defect_counts, total_defects\n• confidence, avg detection confidence, defect_score, base_shelf_life",
    )
    dot.node("end", "End", shape="oval", fillcolor="#E3F2FD")

    dot.edge("start", "inputs")
    dot.edge("inputs", "base")
    dot.edge("base", "loop")
    dot.edge("loop", "score")
    dot.edge("score", "loop", label="More defects?", fontsize="10")
    dot.edge("loop", "metrics", label="All processed", fontsize="10")
    dot.edge("metrics", "band")
    dot.edge("band", "scale")
    dot.edge("scale", "months")
    dot.edge("months", "guard")
    dot.edge("guard", "cap", label="Yes", fontsize="10")
    dot.edge("cap", "days")
    dot.edge("guard", "days", label="No", fontsize="10")
    dot.edge("days", "confidence")
    dot.edge("confidence", "conf_guard")
    dot.edge("conf_guard", "uncertain", label="Yes", fontsize="10")
    dot.edge("uncertain", "category")
    dot.edge("conf_guard", "category", label="No", fontsize="10")
    dot.edge("category", "output")
    dot.edge("output", "end")

    return dot


def main() -> None:
    dot = build_flowchart()
    output_dir = Path(__file__).parent
    base_path = output_dir / "shelf_life_flowchart"
    dot.save(filename=str(base_path))
    try:
        dot.render(filename=str(base_path), cleanup=True)
        print(f"Flowchart rendered to {base_path.with_suffix('.png')}")
    except Exception as exc:  # pragma: no cover - rendering optional
        print(f"Graphviz render skipped ({exc}). Source saved to {base_path.with_suffix('.gv')}")


if __name__ == "__main__":
    main()

