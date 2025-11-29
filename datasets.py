# datasets.py
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import nltk

from kaggle_utils import download_kaggle_dataset


TaskData = Tuple[List[str], List[int]]


def load_sarcasm_dataset(max_samples: int = 2000) -> TaskData:
    task_dir = download_kaggle_dataset("sarcasm_kaggle")
    # common file name: "Sarcasm_Headlines_Dataset_v2.json"
    json_files = list(task_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError("Sarcasm dataset JSON file not found in Kaggle folder.")

    df = pd.read_json(json_files[0], lines=True)
    df = df.sample(frac=1.0, random_state=42).head(max_samples)
    texts = df["headline"].astype(str).tolist()
    labels = df["is_sarcastic"].astype(int).tolist()
    return texts, labels


def load_fake_news_dataset(max_samples: int = 2000) -> TaskData:
    task_dir = download_kaggle_dataset("fake_news")
    # dataset has "Fake.csv" and "True.csv"
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
    texts = df["title"].fillna("") + " " + df["text"].fillna("")
    return texts.tolist(), df["label"].astype(int).tolist()


def load_amazon_reviews_dataset(max_samples: int = 2000) -> TaskData:
    """
    Kaggle dataset: bittlingmayer/amazonreviews
    Files like train.ft.txt / test.ft.txt in fastText format:
      __label__1 This is a bad review...
      __label__2 This is a good review...
    We'll parse the first *.ft.txt file we find.
    """
    task_dir = download_kaggle_dataset("amazon_reviews")

    # Prefer fastText files
    ft_files = list(task_dir.glob("*.ft.txt"))
    txt_files = list(task_dir.glob("*.txt")) if not ft_files else []
    candidates = ft_files or txt_files
    if not candidates:
        raise RuntimeError(
            "No fastText .ft.txt or .txt files found in amazon reviews dataset."
        )

    data_path = candidates[0]
    texts: List[str] = []
    labels: List[int] = []

    with open(data_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # collect leading __label__ tokens
            label_tokens = []
            i = 0
            while i < len(parts) and parts[i].startswith("__label__"):
                label_tokens.append(parts[i])
                i += 1
            if not label_tokens or i >= len(parts):
                continue  # malformed line

            # fastText uses __label__1 / __label__2 (1 = neg, 2 = pos)
            first_label = label_tokens[0]
            if first_label.endswith("1"):
                label = 0
            elif first_label.endswith("2"):
                label = 1
            else:
                # fall back: even/odd as 0/1
                try:
                    num = int(first_label.split("__label__")[-1])
                    label = int(num % 2 == 1)
                except Exception:
                    continue

            text = " ".join(parts[i:])
            if not text:
                continue

            texts.append(text)
            labels.append(label)
            if len(texts) >= max_samples:
                break

    if not texts:
        raise RuntimeError("Parsed 0 examples from Amazon reviews fastText file.")

    return texts, labels



def load_hate_speech_dataset(max_samples: int = 2000) -> TaskData:
    task_dir = download_kaggle_dataset("hate_speech")
    csv_files = list(task_dir.glob("*.csv"))
    if not csv_files:
        raise RuntimeError("Hate speech CSV not found.")
    df = pd.read_csv(csv_files[0])
    # many variants; assume columns 'tweet' and 'class' or similar
    text_col = "tweet" if "tweet" in df.columns else df.columns[-1]
    label_col = "class" if "class" in df.columns else df.columns[0]

    df = df.sample(frac=1.0, random_state=42).head(max_samples)
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].astype(int).tolist()
    return texts, labels


def ensure_nltk_treebank():
    try:
        nltk.data.find("corpora/treebank")
    except LookupError:
        nltk.download("treebank")
        nltk.download("universal_tagset")


def load_pos_treebank_dataset(max_samples: int = 2000) -> TaskData:
    """
    Create a simple POS tagging-style classification dataset:
    each sample = one sentence; label = majority POS tag category index.
    Not perfect, but enough to have a 'structural' probing task.
    """
    ensure_nltk_treebank()
    from nltk.corpus import treebank

    tagged_sents = treebank.tagged_sents(tagset="universal")[: max_samples]
    texts = []
    labels = []

    # map tag to index
    tag_to_id = {}

    for sent in tagged_sents:
        words = [w for (w, t) in sent]
        tags = [t for (w, t) in sent]
        text = " ".join(words)
        # majority tag as label
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
