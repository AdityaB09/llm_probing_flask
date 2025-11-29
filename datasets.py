# datasets.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

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


# ---------- NEW Amazon Reviews: Fine Food (Score 1–2 vs 4–5) ----------

def load_amazon_reviews_dataset(max_samples: int = 2000) -> TaskData:
    """
    Kaggle dataset: snap/amazon-fine-food-reviews

    File: Reviews.csv
    Columns: 'Score' (1-5), 'Text' (review), 'Summary' etc.

    We create a binary label:
        0 = negative (Score <= 2)
        1 = positive (Score >= 4)
    Score == 3 is treated as neutral and dropped.
    """
    task_dir = download_kaggle_dataset("amazon_reviews")

    # Handle both direct and nested locations
    candidates = list(task_dir.rglob("Reviews.csv"))
    if not candidates:
        raise RuntimeError("Reviews.csv not found in Amazon Fine Food dataset.")

    csv_path = candidates[0]
    df = pd.read_csv(csv_path)

    if "Score" not in df.columns or "Text" not in df.columns:
        raise RuntimeError("Expected 'Score' and 'Text' columns in Reviews.csv.")

    # Drop neutral reviews
    df = df[(df["Score"] <= 2) | (df["Score"] >= 4)]

    # Binary label
    labels = (df["Score"] >= 4).astype(int)

    texts = df["Text"].astype(str)
    # prepend summary if present
    if "Summary" in df.columns:
        texts = df["Summary"].fillna("").astype(str) + " : " + texts

    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.sample(frac=1.0, random_state=42).head(max_samples)

    return df["text"].tolist(), df["label"].astype(int).tolist()


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

    tag_to_id = {}
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
    "hate_speech": load_hate_speech_dataset,
    "pos_treebank": load_pos_treebank_dataset,
}
