# history_store.py
from __future__ import annotations
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

HISTORY_PATH = Path("run_history.json")


def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    with open(HISTORY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def append_run(run_record: Dict[str, Any]):
    """
    Adds run_id and timestamp, prepends to history.
    run_record can contain:
      model_name, task_name, max_samples, max_length,
      best_layer, best_f1, best_acc, layer_metrics, ...
    """
    history = load_history()
    if "run_id" not in run_record:
        run_record["run_id"] = str(int(time.time() * 1000))
    run_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
    history.insert(0, run_record)
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
