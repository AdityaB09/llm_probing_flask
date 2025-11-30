from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd
import nltk

from kaggle_utils import download_kaggle_dataset

TaskData = Tuple[List[str], List[int]]


# ---------- Sarcasm ----------

def load_sarcasm_dataset(max_samples: int = 2000) -> TaskData:
    task_dir = download_kaggle_dataset("sarcasm_kaggle")
    json_files = list(task_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError("Sarcasm dataset JSON file not found in Kaggle folder.")

    df = pd.read_json(json_files[0], lines=True)
    df = df.sample(frac=1.0, random_state=42).head(max_samples)
    texts = df["headline"].astype(str).tolist()
    labels = df["is_sarcastic"].astype(int).tolist()
    return texts, labels


# ---------- Fake vs Real News ----------

def load_fake_news_dataset(max_samples: int = 2000) -> TaskData:
    task_dir = download_kaggle_dataset("fake_news")
    fake_path = task_dir / "Fake.csv"
    true_path = task_dir / "True.csv"
    if not fake_path.exists() or not true_path.exists():
        raise RuntimeError("Fake/True CSV files not found in fake-news dataset.")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).head(max_samples)

    texts = (df["title"].fillna("") + " " + df["text"].fillna("")).astype(str)
    return texts.tolist(), df["label"].astype(int).tolist()


# ---------- Amazon Reviews (binary sentiment) ----------

def load_amazon_reviews_dataset(max_samples: int = 2000) -> TaskData:
    """
    Kaggle dataset: nabamitachakraborty/amazon-reviews.

    We try to infer the label/text columns and map sentiment to {0,1}.
    """
    task_dir = download_kaggle_dataset("amazon_reviews")

    csv_files = list(task_dir.rglob("*.csv"))
    if not csv_files:
        raise RuntimeError("No CSV files found in amazon reviews dataset.")

    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)

    # Try to find label + text columns
    label_col = None
    for c in df.columns:
        if c.lower().startswith("label"):
            label_col = c
            break
    if label_col is None:
        raise RuntimeError("Could not find a label column in amazon reviews CSV.")

    text_col = None
    candidates = ["text", "review", "sentence", "comment", "content", "body"]
    lower_map = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in lower_map:
            text_col = lower_map[name]
            break
    if text_col is None:
        text_col = df.columns[1]

    labels_raw = df[label_col].astype(int)
    labels = (labels_raw > labels_raw.min()).astype(int)
    texts = df[text_col].astype(str)

    df_proc = pd.DataFrame({"text": texts, "label": labels})
    df_proc = df_proc.sample(frac=1.0, random_state=42).head(max_samples)

    return df_proc["text"].tolist(), df_proc["label"].astype(int).tolist()


# ---------- IMDB Reviews (binary sentiment) ----------

def load_imdb_reviews_dataset(max_samples: int = 2000) -> TaskData:
    """
    Kaggle dataset: lakshmi25npathi/imdb-dataset-of-50k-movie-reviews.
    """
    task_dir = download_kaggle_dataset("imdb_reviews")
    csv_path = task_dir / "IMDB Dataset.csv"
    if not csv_path.exists():
        csv_files = list(task_dir.rglob("*.csv"))
        if not csv_files:
            raise RuntimeError("No CSV files found in IMDB dataset.")
        csv_path = csv_files[0]

    df = pd.read_csv(csv_path)
    if "review" not in df.columns or "sentiment" not in df.columns:
        raise RuntimeError("Expected 'review' and 'sentiment' columns in IMDB CSV.")

    df = df.sample(frac=1.0, random_state=42).head(max_samples)
    texts = df["review"].astype(str).tolist()
    labels = (df["sentiment"].str.lower() == "positive").astype(int).tolist()
    return texts, labels


# ---------- Hate Speech ----------

def load_hate_speech_dataset(max_samples: int = 2000) -> TaskData:
    task_dir = download_kaggle_dataset("hate_speech")
    csv_files = list(task_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("Hate speech CSV not found.")
    df = pd.read_csv(csv_files[0])

    text_col = "tweet" if "tweet" in df.columns else df.columns[-1]
    label_col = "class" if "class" in df.columns else df.columns[0]

    df = df.sample(frac=1.0, random_state=42).head(max_samples)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


# ---------- POS Structural Task (Treebank) ----------

def ensure_nltk_treebank():
    try:
        nltk.data.find("corpora/treebank")
    except LookupError:
        nltk.download("treebank")
        nltk.download("universal_tagset")


def load_pos_treebank_dataset(max_samples: int = 2000) -> TaskData:
    """
    Simple sentence-level POS majority label.
    """
    ensure_nltk_treebank()
    from nltk.corpus import treebank

    tagged_sents = treebank.tagged_sents(tagset="universal")[: max_samples]
    texts: List[str] = []
    labels: List[int] = []

    tag_to_id: Dict[str, int] = {}
    for sent in tagged_sents:
        words = [w for (w, t) in sent]
        tags = [t for (w, t) in sent]
        text = " ".join(words)
        maj_tag = max(set(tags), key=tags.count)
        if maj_tag not in tag_to_id:
            tag_to_id[maj_tag] = len(tag_to_id)
        label = tag_to_id[maj_tag]
        texts.append(text)
        labels.append(label)

    return texts, labels


TASK_LOADERS = {
    "sarcasm_kaggle": load_sarcasm_dataset,
    "fake_news": load_fake_news_dataset,
    "amazon_reviews": load_amazon_reviews_dataset,
    "imdb_reviews": load_imdb_reviews_dataset,
    "hate_speech": load_hate_speech_dataset,
    "pos_treebank": load_pos_treebank_dataset,
}
