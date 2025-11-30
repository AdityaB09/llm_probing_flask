from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from transformers import AutoModel, AutoTokenizer

from datasets import TASK_LOADERS
from metrics import compute_basic_metrics, expected_calibration_error

# tasks where labels 0/1 mean negative/positive sentiment
SENTIMENT_TASKS = {"amazon_reviews", "imdb_reviews"}


@dataclass
class ProbeConfig:
    model_name: str = "distilbert-base-uncased"
    task_name: str = "sarcasm_kaggle"
    max_samples: int = 1000
    max_length: int = 64
    test_size: float = 0.2
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool_hidden_states(hidden_states: torch.Tensor,
                            attention_mask: torch.Tensor) -> np.ndarray:
    mask = attention_mask.unsqueeze(-1)
    masked = hidden_states * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).cpu().numpy()


def extract_layerwise_representations(config: ProbeConfig,
                                      texts: List[str]) -> List[np.ndarray]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name, output_hidden_states=True)
    model.to(config.device)
    model.eval()

    all_layer_reprs: List[np.ndarray] | None = None
    bs = 16

    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch_texts = texts[i: i + bs]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(config.device) for k, v in enc.items()}
            outputs = model(**enc)
            hidden_states = outputs.hidden_states  # (num_layers+1, B, T, H)

            batch_layer_pools = [
                mean_pool_hidden_states(hs, enc["attention_mask"])
                for hs in hidden_states
            ]

            if all_layer_reprs is None:
                all_layer_reprs = [arr for arr in batch_layer_pools]
            else:
                for layer_idx, arr in enumerate(batch_layer_pools):
                    all_layer_reprs[layer_idx] = np.vstack(
                        [all_layer_reprs[layer_idx], arr]
                    )

    assert all_layer_reprs is not None
    return all_layer_reprs


def _make_probe() -> LogisticRegression:
    return LogisticRegression(
        max_iter=300,      # bumped to avoid convergence warnings
        n_jobs=-1,
        solver="lbfgs",
    )


def run_layerwise_probing(config: ProbeConfig) -> Dict[str, Any]:
    if config.task_name not in TASK_LOADERS:
        raise ValueError(f"Unknown task: {config.task_name}")

    print(f"[PROBING] Loading data for task {config.task_name} ...")
    texts, labels = TASK_LOADERS[config.task_name](max_samples=config.max_samples)
    labels = np.array(labels)
    print(f"[PROBING] Loaded {len(texts)} samples.")

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=labels,
    )

    print("[PROBING] Extracting layerwise representations ...")
    train_layer_reprs = extract_layerwise_representations(config, X_train_texts)
    test_layer_reprs = extract_layerwise_representations(config, X_test_texts)

    num_layers = len(train_layer_reprs)
    print(f"[PROBING] Model has {num_layers} layers of representations.")

    layer_metrics: List[Dict[str, Any]] = []

    for layer_idx in range(num_layers):
        X_train = train_layer_reprs[layer_idx]
        X_test = test_layer_reprs[layer_idx]

        clf = _make_probe()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(X_train, y_train)

        probas = clf.predict_proba(X_test)
        preds = probas.argmax(axis=1)

        m = compute_basic_metrics(y_test, preds)
        ece = expected_calibration_error(y_test, probas)
        layer_metrics.append(
            {
                "layer_index": layer_idx,
                "accuracy": m["accuracy"],
                "f1_weighted": m["f1_weighted"],
                "ece": ece,
            }
        )

    # Determine best layer by F1
    best_layer = max(layer_metrics, key=lambda m: m["f1_weighted"])
    best_layer_index = int(best_layer["layer_index"])

    # Optional: collect example predictions for sentiment tasks
    example_predictions: List[Dict[str, Any]] = []
    if config.task_name in SENTIMENT_TASKS:
        X_train = train_layer_reprs[best_layer_index]
        X_test = test_layer_reprs[best_layer_index]

        clf = _make_probe()
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            clf.fit(X_train, y_train)

        probas = clf.predict_proba(X_test)
        preds = probas.argmax(axis=1)
        confidences = probas.max(axis=1)

        indices = np.arange(len(X_test))
        correct_mask = preds == y_test
        incorrect_mask = ~correct_mask

        def top_k(mask, k):
            idxs = indices[mask]
            if len(idxs) == 0:
                return []
            conf = confidences[mask]
            order = np.argsort(-conf)
            return idxs[order][:k].tolist()

        chosen = top_k(correct_mask, 3) + top_k(incorrect_mask, 3)

        seen = set()
        final_idx: List[int] = []
        for i in chosen:
            if i not in seen:
                seen.add(i)
                final_idx.append(i)
            if len(final_idx) >= 6:
                break

        for i in final_idx:
            example_predictions.append(
                {
                    "text": X_test_texts[i],
                    "gold": int(y_test[i]),
                    "pred": int(preds[i]),
                    "confidence": float(confidences[i]),
                }
            )

    num_classes = int(labels.max() + 1)

    return {
        "config": config.__dict__,
        "task_name": config.task_name,
        "model_name": config.model_name,
        "layer_metrics": layer_metrics,
        "num_layers": num_layers,
        "num_samples": len(texts),
        "num_classes": num_classes,
        "best_layer_index": best_layer_index,
        "example_predictions": example_predictions,
    }
