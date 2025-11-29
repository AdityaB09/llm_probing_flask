# app.py
from __future__ import annotations
from flask import Flask, render_template, request, redirect, url_for, jsonify

from probing_core import ProbeConfig, run_layerwise_probing
from history_store import append_run, load_history


app = Flask(__name__)
app.secret_key = "super-secret-probing-key"  # for WTForms/CSRF if you add forms


AVAILABLE_MODELS = [
    "distilbert-base-uncased",
    "roberta-base",
]

AVAILABLE_TASKS = {
    "sarcasm_kaggle": "Sarcasm Detection (Kaggle)",
    "fake_news": "Fake vs Real News (Kaggle)",
    "amazon_reviews": "Amazon Reviews Sentiment (Kaggle)",
    "hate_speech": "Hate Speech & Offensive Language (Kaggle)",
    "pos_treebank": "POS Structural Task (Treebank)",
}


@app.route("/")
def index():
    history = load_history()[:5]
    return render_template(
        "index.html",
        tasks=AVAILABLE_TASKS,
        models=AVAILABLE_MODELS,
        recent_runs=history,
    )






# Simple JSON endpoint so JS can animate "live" feel
@app.route("/api/history_summary")
def history_summary_api():
    history = load_history()[:10]
    return jsonify(history)

# app.py (only showing changed / new parts)
from flask import Flask, render_template, request, redirect, url_for, jsonify, abort

# ... imports unchanged ...


def build_run_explanation(task_name, model_name, layer_metrics):
    """
    Simple rule-based explanation for the probing result.
    """
    if not layer_metrics:
        return "This run does not have stored layer-wise metrics."

    num_layers = len(layer_metrics)
    best = max(layer_metrics, key=lambda m: m["f1_weighted"])
    shallow = layer_metrics[0]
    mid = layer_metrics[num_layers // 2]
    deep = layer_metrics[-1]

    def fmt(x):
        return f"{x:.3f}"

    lines = []

    lines.append(
        f"For **{model_name}** on **{task_name}**, "
        f"the best F1 was at layer **{best['layer_index']}** "
        f"with F1 = {fmt(best['f1_weighted'])} and accuracy = {fmt(best['accuracy'])}."
    )

    lines.append(
        f"The shallow layer (0) reached F1 = {fmt(shallow['f1_weighted'])}, "
        f"while a mid layer ({mid['layer_index']}) achieved F1 = {fmt(mid['f1_weighted'])}, "
        f"and the deepest layer ({deep['layer_index']}) ended at F1 = {fmt(deep['f1_weighted'])}."
    )

    # crude pattern: where does performance peak?
    if best["layer_index"] <= 1:
        lines.append(
            "Performance peaking at the earliest layers suggests that the task "
            "relies heavily on local lexical features rather than deep abstraction."
        )
    elif best["layer_index"] >= num_layers - 2:
        lines.append(
            "Performance peaking near the last layers indicates that the model "
            "needs deeper semantic representations to solve this task."
        )
    else:
        lines.append(
            "Peak performance in the middle layers is consistent with probing studies "
            "showing that intermediate layers often encode rich syntactic and semantic structure."
        )

    avg_ece = sum(m["ece"] for m in layer_metrics) / len(layer_metrics)
    lines.append(
        f"Across layers, the average calibration error (ECE) is about {fmt(avg_ece)}, "
        "showing how well the probe probabilities align with true correctness."
    )

    return " ".join(lines)


@app.route("/run", methods=["GET", "POST"])
def run_probe_view():
    if request.method == "GET":
        return render_template(
            "run_probe.html",
            tasks=AVAILABLE_TASKS,
            models=AVAILABLE_MODELS,
            error_message=None,
        )

    model_name = request.form.get("model_name", "distilbert-base-uncased")
    task_name = request.form.get("task_name", "sarcasm_kaggle")
    max_samples = int(request.form.get("max_samples", 500))
    max_length = int(request.form.get("max_length", 64))

    config = ProbeConfig(
        model_name=model_name,
        task_name=task_name,
        max_samples=max_samples,
        max_length=max_length,
    )

    try:
        result = run_layerwise_probing(config)
    except Exception as e:
        # Show a friendly error on the same page instead of a raw 500
        app.logger.exception("Probing run failed")
        return render_template(
            "run_probe.html",
            tasks=AVAILABLE_TASKS,
            models=AVAILABLE_MODELS,
            error_message=str(e),
        ), 500

    best_layer = max(result["layer_metrics"], key=lambda m: m["f1_weighted"])

    run_record = {
        "model_name": model_name,
        "task_name": task_name,
        "max_samples": max_samples,
        "max_length": max_length,
        "best_layer": best_layer["layer_index"],
        "best_f1": best_layer["f1_weighted"],
        "best_acc": best_layer["accuracy"],
        "layer_metrics": result["layer_metrics"],
    }
    append_run(run_record)

    explanation = build_run_explanation(
        task_name, model_name, result["layer_metrics"]
    )

    return render_template(
        "results.html",
        task_label=AVAILABLE_TASKS.get(task_name, task_name),
        model_name=model_name,
        result=result,
        run_record=run_record,
        explanation=explanation,
    )


@app.route("/history")
def history_view():
    history = load_history()
    return render_template("history.html", history=history)


@app.route("/runs/<run_id>")
def run_detail(run_id):
    history = load_history()
    record = next((r for r in history if r.get("run_id") == run_id), None)
    if record is None:
        abort(404)

    layer_metrics = record.get("layer_metrics", [])
    explanation = build_run_explanation(
        record.get("task_name"), record.get("model_name"), layer_metrics
    )

    return render_template(
        "run_detail.html",
        run=record,
        explanation=explanation,
        task_label=AVAILABLE_TASKS.get(record.get("task_name"), record.get("task_name")),
    )


if __name__ == "__main__":
    app.run(debug=True)
