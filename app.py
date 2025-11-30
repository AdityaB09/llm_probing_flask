from __future__ import annotations

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    abort,
    redirect,
    url_for,
    Response,
)

from probing_core import ProbeConfig, run_layerwise_probing
from history_store import append_run, load_history, get_run, update_run_note
from summarizer import generate_run_summary
from report_utils import build_markdown_report
from datasets import TASK_LOADERS

app = Flask(__name__)
app.secret_key = "super-secret-probing-key"

# --- Models & Tasks ----------------------------------------------------------

AVAILABLE_MODELS = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "distilroberta-base",
    "microsoft/deberta-v3-base",
]

AVAILABLE_TASKS = {
    "sarcasm_kaggle": "Sarcasm Detection (Kaggle)",
    "fake_news": "Fake vs Real News (Kaggle)",
    "amazon_reviews": "Amazon Reviews Sentiment (Kaggle)",
    "imdb_reviews": "IMDB Movie Reviews Sentiment (Kaggle)",
    "hate_speech": "Hate Speech & Offensive Language (Kaggle)",
    "pos_treebank": "POS Structural Task (Treebank)",
}

TASK_BLURBS = {
    "amazon_reviews": (
        "Each example is an Amazon product review with a binary sentiment label "
        "(negative vs positive). We probe where overall product sentiment "
        "becomes linearly separable in the model."
    ),
    "imdb_reviews": (
        "Each example is a full IMDB movie review labeled as positive or negative. "
        "Reviews are long and complex, so deeper layers often matter more."
    ),
    "sarcasm_kaggle": (
        "News headlines annotated for sarcasm — a task that mixes surface cues "
        "with subtle world knowledge."
    ),
    "fake_news": (
        "News articles labeled as fake or real, probing whether layers encode "
        "credibility and stylistic signals."
    ),
    "hate_speech": (
        "Tweets labeled for hate/offensive content, focusing on social-safety "
        "and toxicity patterns."
    ),
    "pos_treebank": (
        "Treebank sentences summarized by their dominant POS tag, turning "
        "syntactic structure into a probing target."
    ),
}


# --- Helpers -----------------------------------------------------------------


def _build_radar_data(run: dict) -> dict:
    lm = run.get("layer_metrics", [])
    if not lm:
        return {"labels": [], "values": []}

    num_layers = run.get("num_layers", len(lm))
    best = max(lm, key=lambda m: m.get("f1_weighted", 0.0))
    best_layer = int(best.get("layer_index", best.get("layer", 0)))
    best_f1 = float(best.get("f1_weighted", 0.0))
    best_acc = float(best.get("accuracy", 0.0))

    eces = [float(m.get("ece", 0.0)) for m in lm]
    avg_ece = sum(eces) / len(eces)
    var_ece = sum((e - avg_ece) ** 2 for e in eces) / len(eces)

    if num_layers > 1:
        depth_norm = best_layer / (num_layers - 1)
    else:
        depth_norm = 0.0

    return {
        "labels": [
            "Best F1",
            "Best accuracy",
            "Normalized depth",
            "Avg ECE",
            "ECE variance",
        ],
        "values": [
            best_f1,
            best_acc,
            depth_norm,
            max(0.0, 1.0 - avg_ece),  # invert so "better" is higher
            max(0.0, 1.0 - var_ece),
        ],
    }


# --- Routes ------------------------------------------------------------------


@app.route("/")
def index():
    history = load_history()[:5]
    return render_template(
        "index.html",
        tasks=AVAILABLE_TASKS,
        models=AVAILABLE_MODELS,
        recent_runs=history,
    )


@app.route("/run", methods=["GET", "POST"])
def run_probe_view():
    if request.method == "GET":
        preselected_task = request.args.get("task")
        preselected_model = request.args.get("model")
        return render_template(
            "run_probe.html",
            tasks=AVAILABLE_TASKS,
            models=AVAILABLE_MODELS,
            preselected_task=preselected_task,
            preselected_model=preselected_model,
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
        app.logger.exception("Probing run failed")
        return render_template(
            "run_probe.html",
            tasks=AVAILABLE_TASKS,
            models=AVAILABLE_MODELS,
            preselected_task=task_name,
            preselected_model=model_name,
            error_message=str(e),
        ), 500

    best_layer = max(result["layer_metrics"], key=lambda m: m["f1_weighted"])

    run_record = {
        "model_name": model_name,
        "task_name": task_name,
        "task_display_name": AVAILABLE_TASKS.get(task_name, task_name),
        "max_samples": max_samples,
        "max_length": max_length,
        "num_samples": result["num_samples"],
        "num_layers": result["num_layers"],
        "num_classes": result["num_classes"],
        "best_layer": result["best_layer_index"],
        "best_f1": best_layer["f1_weighted"],
        "best_acc": best_layer["accuracy"],
        "layer_metrics": result["layer_metrics"],
        "example_predictions": result["example_predictions"],
    }
    run_id = append_run(run_record)

    # Prepare data for this immediate results page
    run_record["run_id"] = run_id
    radar_data = _build_radar_data(run_record)
    summary = generate_run_summary(run_record)

    return render_template(
        "results.html",
        task_label=run_record["task_display_name"],
        model_name=model_name,
        result=result,
        run_record=run_record,
        explanation=summary,
        example_predictions=result["example_predictions"],
        task_name=task_name,
        radar_data=radar_data,
    )


@app.route("/history")
def history_view():
    history = load_history()
    return render_template("history.html", history=history, tasks=AVAILABLE_TASKS)


@app.route("/runs/<run_id>")
def run_detail(run_id):
    record = get_run(run_id)
    if record is None:
        abort(404)

    radar_data = _build_radar_data(record)
    summary = generate_run_summary(record)

    return render_template(
        "run_detail.html",
        run=record,
        explanation=summary,
        task_label=record.get("task_display_name", AVAILABLE_TASKS.get(
            record.get("task_name"), record.get("task_name")
        )),
        radar_data=radar_data,
    )


@app.route("/runs/<run_id>/note", methods=["POST"])
def add_note(run_id):
    note = request.form.get("note", "")
    update_run_note(run_id, note)
    return redirect(url_for("run_detail", run_id=run_id))


@app.route("/runs/<run_id>/report")
def download_report(run_id):
    record = get_run(run_id)
    if record is None:
        abort(404)
    md = build_markdown_report(record)
    filename = f"probe_report_{run_id}.md"
    return Response(
        md,
        mimetype="text/markdown",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.route("/compare", methods=["GET", "POST"])
def compare_view():
    if request.method == "GET":
        return render_template(
            "compare.html",
            tasks=AVAILABLE_TASKS,
            models=AVAILABLE_MODELS,
            comparison=None,
            error_message=None,
        )

    task_name = request.form.get("task_name")
    model_a = request.form.get("model_a")
    model_b = request.form.get("model_b")
    max_samples = int(request.form.get("max_samples", 500))
    max_length = int(request.form.get("max_length", 64))

    if model_a == model_b:
        return render_template(
            "compare.html",
            tasks=AVAILABLE_TASKS,
            models=AVAILABLE_MODELS,
            comparison=None,
            error_message="Please choose two different models.",
        )

    try:
        config_a = ProbeConfig(
            model_name=model_a,
            task_name=task_name,
            max_samples=max_samples,
            max_length=max_length,
        )
        config_b = ProbeConfig(
            model_name=model_b,
            task_name=task_name,
            max_samples=max_samples,
            max_length=max_length,
        )
        result_a = run_layerwise_probing(config_a)
        result_b = run_layerwise_probing(config_b)
    except Exception as e:
        app.logger.exception("Model comparison failed")
        return render_template(
            "compare.html",
            tasks=AVAILABLE_TASKS,
            models=AVAILABLE_MODELS,
            comparison=None,
            error_message=str(e),
        ), 500

    comp = {
        "task_name": task_name,
        "task_label": AVAILABLE_TASKS.get(task_name, task_name),
        "model_a": model_a,
        "model_b": model_b,
        "metrics_a": result_a["layer_metrics"],
        "metrics_b": result_b["layer_metrics"],
    }

    return render_template(
        "compare.html",
        tasks=AVAILABLE_TASKS,
        models=AVAILABLE_MODELS,
        comparison=comp,
        error_message=None,
    )


@app.route("/explore")
def explore_view():
    task_name = request.args.get("task") or "sarcasm_kaggle"
    if task_name not in TASK_LOADERS:
        abort(404)

    # Sample once per request (keep N small to stay fast)
    texts, labels = TASK_LOADERS[task_name](max_samples=1000)
    label_counts = {}
    for y in labels:
        label_counts[int(y)] = label_counts.get(int(y), 0) + 1

    # Simple length histogram (in tokens split by space)
    lengths = [len(t.split()) for t in texts]
    bins = [0, 10, 20, 40, 60, 80, 120, 200]
    counts = [0] * (len(bins) - 1)
    for L in lengths:
        for i in range(len(bins) - 1):
            if bins[i] <= L < bins[i + 1]:
                counts[i] += 1
                break

    length_hist = {
        "bins": [f"{bins[i]}–{bins[i+1]-1}" for i in range(len(bins) - 1)],
        "counts": counts,
    }

    # Show a few random examples
    examples = []
    for t, y in zip(texts[:10], labels[:10]):
        examples.append({"text": t, "label": int(y)})

    return render_template(
        "explorer.html",
        tasks=AVAILABLE_TASKS,
        current_task=task_name,
        task_label=AVAILABLE_TASKS.get(task_name, task_name),
        label_counts=label_counts,
        length_hist=length_hist,
        examples=examples,
    )


@app.route("/api/history_summary")
def history_summary_api():
    history = load_history()[:10]
    return jsonify(history)


if __name__ == "__main__":
    app.run(debug=True)
