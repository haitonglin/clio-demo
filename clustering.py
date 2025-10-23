"""
clustering.py
---------------------------------
Clio-style clustering for extracted facets.

Pipeline:
1) Embed the request facet using all-mpnet-base-v2 (768-d embeddings).
2) Run K-Means on embedding space.
3) Return cluster labels and summary statistics.

This implementation matches the Clio paper's description:
- "We first embed each extracted summary using all-mpnet-base-v2...
   We then generate base-level clusters by running k-means in embedding space."
"""

from __future__ import annotations

import pandas as pd
import math
from typing import Optional, Tuple
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

# -----------------------------------------------------
# 1) Embedding
# -----------------------------------------------------

def embed_facets(df: pd.DataFrame, text_column: str = "request") -> Tuple[pd.DataFrame, list]:
    """
    Embeds the request facet using all-mpnet-base-v2, returning both:
      - original DataFrame (unaltered)
      - embedding matrix (list of 768-d vectors)

    df must contain the column: text_column
    """
    if text_column not in df.columns:
        raise ValueError(f"Expected column '{text_column}' in dataframe.")

    model = SentenceTransformer("all-mpnet-base-v2")
    texts = df[text_column].fillna("").tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    return df, embeddings


# -----------------------------------------------------
# 2) K-Means clustering
# -----------------------------------------------------

def choose_k(n_samples: int, k_min: int = 3, k_max: int = 40) -> int:
    """
    Heuristic for selecting k based on dataset size.
    - Scales k with n, but keeps it within bounds.
    - Default caps: 3 ≤ k ≤ 40
    - Users may override k by passing a value to run_kmeans()

    Example behavior:
      n=100   => k=3-5
      n=300   => k~6
      n=1000  => k~20
      n>=2000 => k<=40
    """
    # proportional rule
    raw_k = max(k_min, n_samples // 50)

    # clip to range
    return min(raw_k, k_max)


def run_kmeans(embeddings, k: Optional[int] = None, random_state: int = 42):
    """
    Run K-Means on embeddings and return cluster labels.
    If k is None, choose using heuristic based on dataset size.
    """
    n = len(embeddings)
    if k is None:
        k = choose_k(n)

    km = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
    labels = km.fit_predict(embeddings)
    return labels, k


# -----------------------------------------------------
# 3) Add labels to dataframe
# -----------------------------------------------------

def assign_clusters(df: pd.DataFrame, labels) -> pd.DataFrame:
    """
    Append cluster_id column to dataframe.
    """
    df = df.copy()
    df["cluster_id"] = labels
    return df


# -----------------------------------------------------
# 4) Simple cluster summary (counts per cluster)
# -----------------------------------------------------

def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns cluster size summary: cluster_id, count
    """
    return (
        df.groupby("cluster_id")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
