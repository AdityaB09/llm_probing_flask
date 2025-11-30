from __future__ import annotations
from typing import Dict, List
from summarizer import generate_run_summary


def build_markdown_report(run: Dict) -> str:
    """
    Build a markdown report for a single run.
    """
    summary = generate_run_summary(run)
    lm: List[Dict] = run.get("layer_metrics", [])

    lines = []
    lines.append(f"# LLM Probing Report")
    lines.append("")
    lines.append(f"**Model:** `{run.get('model_name')}`")
    lines.append(f"**Task:** {run.get('task_display_name', run.get('task_name'))}")
    lines.append(f"**Run ID:** `{run.get('run_id')}`")
    lines.append(f"**Timestamp:** {run.get('timestamp')}")
    lines.append("")
    lines.append("## High-level summary")
    lines.append("")
    lines.append(summary)
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- Max samples: {run.get('max_samples')}")
    lines.append(f"- Max sequence length: {run.get('max_length')}")
    lines.append("")
    lines.append("## Layerwise metrics")
    lines.append("")
    lines.append("| Layer | Accuracy | Weighted F1 | ECE |")
    lines.append("|-------|----------|-------------|-----|")
    for m in lm:
        layer = m.get("layer_index", m.get("layer"))
        acc = m.get("accuracy", 0.0)
        f1 = m.get("f1_weighted", 0.0)
        ece = m.get("ece", 0.0)
        lines.append(f"| {layer} | {acc:.3f} | {f1:.3f} | {ece:.3f} |")

    if run.get("note"):
        lines.append("")
        lines.append("## Analyst note")
        lines.append("")
        lines.append(run["note"])

    return "\n".join(lines)
