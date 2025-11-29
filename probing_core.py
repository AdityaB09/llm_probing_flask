# probing_core.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from transformers import AutoModel, AutoTokenizer

from datasets import TASK_LOADERS
from metrics import compute_basic_metrics, expected_calibration_error


@dataclass
class ProbeConfig:
    model_name: str = "distilbert-base-uncased"
    task_name: str = "sarcasm_kaggle"
    max_samples: int = 1000
    max_length: int = 64
    test_size: float = 0.2
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def mean_pool_hidden_states(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> np.ndarray:
    """
    hidden_states: (batch, seq_len, hidden_dim)
    attention_mask: (batch, seq_len)
    returns: (batch, hidden_dim)
    """
    mask = attention_mask.unsqueeze(-1)  # (b, seq, 1)
    masked = hidden_states * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).cpu().numpy()


def extract_layerwise_representations(config: ProbeConfig, texts: List[str]):
    """
    Returns:
      layer_reprs: list of np.ndarray of shape (N, hidden_dim) for each layer
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModel.from_pretrained(config.model_name, output_hidden_states=True)
    model.to(config.device)
    model.eval()

    all_layer_reprs = None

    bs = 16
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            batch_texts = texts[i : i + bs]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(config.device) for k, v in enc.items()}
            outputs = model(**enc)
            hidden_states = outputs.hidden_states  # tuple(len_layers) of (b, seq, hid)

            batch_layer_pools = [
                mean_pool_hidden_states(hs, enc["attention_mask"])
                for hs in hidden_states
            ]  # list of (b, hid) np arrays

            if all_layer_reprs is None:
                all_layer_reprs = [arr for arr in batch_layer_pools]
            else:
                for layer_idx, arr in enumerate(batch_layer_pools):
                    all_layer_reprs[layer_idx] = np.vstack([all_layer_reprs[layer_idx], arr])

    return all_layer_reprs  # list length = n_layers


def run_layerwise_probing(config: ProbeConfig) -> Dict[str, Any]:
    """
    Main pipeline used by the Flask app.
    Returns a dict with:
      - config
      - layer_metrics: list of dicts per layer
      - num_layers, num_samples, num_classes
    """
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

    layer_metrics = []

    for layer_idx in range(num_layers):
        X_train = train_layer_reprs[layer_idx]
        X_test = test_layer_reprs[layer_idx]

        clf = LogisticRegression(
            max_iter=200,
            n_jobs=-1,
            multi_class="auto",
            class_weight="balanced",
        )
        clf.fit(X_train, y_train)

        probas = clf.predict_proba(X_test)
        preds = probas.argmax(axis=1)

        m = compute_basic_metrics(y_test, preds)
        ece = expected_calibration_error(y_test, probas)
        record = {
            "layer_index": layer_idx,
            "accuracy": m["accuracy"],
            "f1_weighted": m["f1_weighted"],
            "ece": ece,
        }
        layer_metrics.append(record)

    num_classes = int(labels.max() + 1)

    return {
        "config": config.__dict__,
        "layer_metrics": layer_metrics,
        "num_layers": num_layers,
        "num_samples": len(texts),
        "num_classes": num_classes,
    }
