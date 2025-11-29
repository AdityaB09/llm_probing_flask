# kaggle_utils.py
import os
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

# Task -> Kaggle dataset id
KAGGLE_DATASETS = {
    "sarcasm_kaggle": "rmisra/news-headlines-dataset-for-sarcasm-detection",
    "fake_news": "clmentbisaillon/fake-and-real-news-dataset",
    # switched to Fine Food Reviews for Amazon sentiment
    "amazon_reviews": "snap/amazon-fine-food-reviews",
    "hate_speech": "mrmorj/hate-speech-and-offensive-language-dataset",
}


def get_data_root() -> Path:
    root = Path("data")
    root.mkdir(exist_ok=True)
    return root


def ensure_kaggle_auth():
    """
    Makes sure Kaggle is configured. Raises a helpful error if not.
    """
    has_file = Path("~/.kaggle/kaggle.json").expanduser().exists()
    has_env = "KAGGLE_USERNAME" in os.environ and "KAGGLE_KEY" in os.environ
    if not (has_file or has_env):
        raise RuntimeError(
            "Kaggle API is not configured. "
            "Create ~/.kaggle/kaggle.json or set KAGGLE_USERNAME and KAGGLE_KEY."
        )


def download_kaggle_dataset(task_key: str) -> Path:
    """
    Download & unzip dataset for a known task key. Returns the local folder path.
    """
    if task_key not in KAGGLE_DATASETS:
        raise ValueError(f"Unknown Kaggle task key: {task_key}")

    ensure_kaggle_auth()

    dataset_id = KAGGLE_DATASETS[task_key]
    data_root = get_data_root()
    task_dir = data_root / task_key
    task_dir.mkdir(exist_ok=True, parents=True)

    api = KaggleApi()
    api.authenticate()

    # Download only if folder is empty
    if not any(task_dir.iterdir()):
        print(f"[KAGGLE] Downloading {dataset_id} into {task_dir} ...")
        api.dataset_download_files(dataset_id, path=task_dir, unzip=True)
        print("[KAGGLE] Download complete.")
    else:
        print(f"[KAGGLE] Using cached dataset at {task_dir}")

    return task_dir
