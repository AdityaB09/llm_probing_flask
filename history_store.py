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


def _save_history(history: List[Dict[str, Any]]) -> None:
    with open(HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def append_run(run_record: Dict[str, Any]) -> str:
    history = load_history()
    if "run_id" not in run_record:
        run_record["run_id"] = str(int(time.time() * 1000))
    run_record["timestamp"] = datetime.utcnow().isoformat() + "Z"
    history.insert(0, run_record)
    _save_history(history)
    return run_record["run_id"]


def get_run(run_id: str) -> Dict[str, Any] | None:
    history = load_history()
    for r in history:
        if r.get("run_id") == run_id:
            return r
    return None


def update_run_note(run_id: str, note: str) -> None:
    history = load_history()
    for r in history:
        if r.get("run_id") == run_id:
            r["note"] = note.strip()
            break
    _save_history(history)
