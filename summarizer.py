from __future__ import annotations

from typing import Dict, List
import numpy as np


def _depth_phrase(best_layer: int, num_layers: int) -> str:
  if num_layers <= 1:
    return "the single available layer"
  if best_layer >= num_layers - 2:
    return "the upper (task-specific) layers"
  if best_layer <= 1:
    return "the very first layers, close to the input embeddings"
  return "the middle layers"


def _calibration_phrase(ece_values: List[float]) -> str:
  if not ece_values:
    return "Calibration could not be computed."

  mean_ece = float(np.mean(ece_values))
  if mean_ece < 0.03:
    return f"Overall calibration is excellent (mean ECE ≈ {mean_ece:.3f}), so probability estimates are very trustworthy."
  if mean_ece < 0.08:
    return f"Calibration is reasonable (mean ECE ≈ {mean_ece:.3f}); probabilities are usable but could be improved with temperature scaling."
  return f"Calibration is relatively weak (mean ECE ≈ {mean_ece:.3f}), suggesting that logits are over- or under-confident."


def generate_run_summary(run: Dict) -> str:
  model = run.get("model_name", "Unknown model")
  task_name = run.get("task_display_name", run.get("task_name", "unknown task"))
  num_samples = run.get("num_samples", 0)
  num_layers = run.get("num_layers", len(run.get("layer_metrics", [])))
  num_classes = run.get("num_classes", 0)

  layer_metrics = run.get("layer_metrics", [])
  if not layer_metrics:
    return (
      f"This run used {model} on the task “{task_name}”, "
      f"but no layer-wise metrics were recorded, so we cannot summarise performance."
    )

  for m in layer_metrics:
    if "layer_index" in m and "layer" not in m:
      m["layer"] = m["layer_index"]

  best = max(layer_metrics, key=lambda m: m.get("f1_weighted", 0.0))
  best_layer = int(best.get("layer", 0))
  best_f1 = float(best.get("f1_weighted", 0.0))
  best_acc = float(best.get("accuracy", 0.0))

  ece_values = [float(m.get("ece", 0.0)) for m in layer_metrics]
  calib_phrase = _calibration_phrase(ece_values)
  depth_phrase = _depth_phrase(best_layer, num_layers)

  depth_sentence = (
    f"The most linearly separable representation for this task appears around layer {best_layer}, "
    f"where weighted F1 reaches {best_f1:.3f} and accuracy reaches {best_acc:.3f}. "
    f"This suggests that {depth_phrase} encode the most task-specific signal."
  )

  if num_classes == 2:
    setup_sentence = (
      f"We probed {model} on the binary classification task “{task_name}” using {num_samples} examples "
      f"and {num_layers} transformer layers."
    )
  else:
    setup_sentence = (
      f"We probed {model} on the {num_classes}-way classification task “{task_name}” "
      f"using {num_samples} examples across {num_layers} transformer layers."
    )

  f1_first = layer_metrics[0].get("f1_weighted", 0.0)
  f1_last = layer_metrics[-1].get("f1_weighted", 0.0)
  if f1_last > f1_first + 0.05:
    trend_sentence = (
      "Performance improves noticeably from the embedding layers to the final layers, "
      "indicating that the model gradually builds up task-specific abstractions."
    )
  elif f1_last < f1_first - 0.05:
    trend_sentence = (
      "Interestingly, earlier layers perform better than the final layers, "
      "which may indicate that the task aligns more with generic lexical/structural cues than with the final classifier head."
    )
  else:
    trend_sentence = (
      "Performance remains relatively stable across depth, suggesting that the information needed for this task "
      "is distributed rather than concentrated in a single block."
    )

  return " ".join([setup_sentence, depth_sentence, calib_phrase, trend_sentence])
