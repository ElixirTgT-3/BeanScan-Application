"""Generate a defect-classification/severity flowchart using Graphviz.

Usage:
    python tools/defect_severity_flowchart.py

The script produces two files in the tools/ directory:
    - defect_severity_flowchart.gv (Graphviz source)
    - defect_severity_flowchart.png (rendered diagram, if Graphviz is installed)
"""

from pathlib import Path

try:
    from graphviz import Digraph
except ImportError as exc:  # pragma: no cover - guidance for local execution
    raise SystemExit(
        "graphviz Python package is required. Install with `pip install graphviz` "
        "and ensure Graphviz binaries are available on PATH."
    ) from exc


def build_flowchart() -> Digraph:
    dot = Digraph("DefectSeverityFlow", format="png")
    dot.attr(rankdir="TB", fontsize="12", fontname="Helvetica")
    dot.attr("node", shape="rectangle", style="rounded,filled", fillcolor="#F7F5F2")
    dot.attr("edge", arrowsize="0.8")

    dot.node("start", "Start", shape="oval", fillcolor="#E6F4EA")
    dot.node("capture", "Capture / import bean image")
    dot.node("preprocess", "Preprocess image\nResize 224×224 · Normalize pixels")
    dot.node("fastercnn", "Faster R-CNN inference\n(MobileNetV3 backbone)", shape="rectangle", fillcolor="#E8F0FE")
    dot.node("loop", "Iterate each predicted box", shape="diamond", fillcolor="#FFF3CD")
    dot.node("conf_check", "Confidence ≥ 0.5?", shape="diamond", fillcolor="#FFF3CD")
    dot.node("discard", "Discard detection", shape="rectangle", fillcolor="#FDECEA")
    dot.node("keep", "Normalize detection data\n• defect_type\n• confidence\n• bbox / coordinates", fillcolor="#F1F8E9")
    dot.node("accumulate", "Accumulate defect sequence\n(type, confidence, count)")
    dot.node("score", "Compute weighted defect score\nΣ(weight × confidence × count)")
    dot.node("percentage", "Defect percentage = clamp(min(score/45, 1.5) × 100)")
    dot.node("severity", "Select severity band\n<mild · moderate · severe>")
    dot.node("summary", "Build defect summary\n(severity, counts, avg confidence, %)")
    dot.node("store", "Store detections + summary\n(e.g., Supabase DEFECT table)")
    dot.node("display", "Display defect types and severity in app")
    dot.node("end", "End", shape="oval", fillcolor="#E6F4EA")

    dot.edges([
        ("start", "capture"),
        ("capture", "preprocess"),
        ("preprocess", "fastercnn"),
        ("fastercnn", "loop"),
        ("loop", "conf_check"),
    ])

    dot.edge("conf_check", "discard", label="No", fontsize="10", fontname="Helvetica")
    dot.edge("discard", "loop")
    dot.edge("conf_check", "keep", label="Yes", fontsize="10", fontname="Helvetica")
    dot.edge("keep", "accumulate")
    dot.edge("accumulate", "loop", label="More boxes?", fontsize="10", fontname="Helvetica")

    dot.edge("loop", "score", label="All processed", fontsize="10", fontname="Helvetica")
    dot.edge("score", "percentage")
    dot.edge("percentage", "severity")
    dot.edge("severity", "summary")
    dot.edge("summary", "store")
    dot.edge("store", "display")
    dot.edge("display", "end")

    return dot


def main() -> None:
    dot = build_flowchart()
    output_dir = Path(__file__).parent
    source_path = output_dir / "defect_severity_flowchart"
    dot.save(filename=str(source_path))
    try:
        dot.render(filename=str(source_path), cleanup=True)
        print(f"Flowchart rendered to {source_path.with_suffix('.png')}")
    except Exception as exc:  # pragma: no cover - rendering optional
        print(f"Graphviz render skipped ({exc}). Source saved to {source_path.with_suffix('.gv')}")


if __name__ == "__main__":
    main()

