"""
hierarchizer_demo.py
====================
Lightweight hierarchy builder for demos with few clusters.
This version groups clusters into parent groups using centroid similarity
(agglomerative clustering over cluster centroids). No LLM naming in this demo.

Public API:
- HierParams
- build_hierarchy(df, embeddings, params)

Expected df columns:
- params.cluster_col (default="cluster_id")
- params.text_col    (default="request")   # included for future parent naming

Returns a JSON-friendly dict and optionally saves hierarchy.json.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path
import json
import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances


# =========================
# Config / parameters
# =========================

@dataclass
class HierParams:
    text_col: str = "request"
    cluster_col: str = "cluster_id"

    # For this demo, we keep hierarchy shallow. With few clusters, 1 parent is fine.
    n_parents: Optional[int] = None
    distance_threshold: Optional[float] = None
    linkage: str = "average"

    random_state: int = 42
    save_dir: Optional[str] = "artifacts/hierarchy"


# =========================
# Helpers
# =========================

def _cluster_ids_sorted(arr) -> List[int]:
    """Convert cluster IDs to sorted ints."""
    return sorted(list(map(int, arr)))


# =========================
# Core functions
# =========================

def compute_cluster_centroids(df: pd.DataFrame, embeddings: np.ndarray, params: HierParams):
    """
    Compute centroid embedding per cluster (mean of item embeddings).
    """
    ccol = params.cluster_col
    if ccol not in df.columns:
        raise ValueError(f"df must contain column '{ccol}'")
    if len(df) != embeddings.shape[0]:
        raise ValueError("len(df) must match embeddings.shape[0] (row alignment required)")

    cluster_ids = _cluster_ids_sorted(pd.to_numeric(df[ccol], errors="coerce").dropna().unique())
    centroids = []
    for cid in cluster_ids:
        mask = (df[ccol].astype(int).values == cid)
        vecs = embeddings[mask]
        if vecs.shape[0] == 0:
            centroids.append(np.zeros((embeddings.shape[1],), dtype=float))
        else:
            centroids.append(vecs.mean(axis=0))
    return np.vstack(centroids), cluster_ids


def group_clusters(centroids: np.ndarray, params: HierParams):
    """
    Agglomerative clustering over centroids.

    For tiny demos, if both n_parents and distance_threshold are None,
    we return a single parent grouping (all siblings under one root).
    """
    K = centroids.shape[0]
    if K <= 1:
        return np.zeros((K,), dtype=int)

    # If demo and user didn't configure thresholds, put everything under one parent
    if params.n_parents is None and params.distance_threshold is None:
        return np.zeros((K,), dtype=int)

    # Otherwise run agglomerative clustering
    D = cosine_distances(centroids)

    kwargs = dict(
        n_clusters=params.n_parents,
        distance_threshold=params.distance_threshold,
        linkage=params.linkage,
        affinity="precomputed"
    )
    # sklearn rules: one of these must be removed when the other is set
    if params.n_parents is None:
        kwargs.pop("n_clusters")
    if params.distance_threshold is None:
        kwargs.pop("distance_threshold")

    model = AgglomerativeClustering(**kwargs)
    return model.fit_predict(D)


def build_hierarchy(df: pd.DataFrame, embeddings: np.ndarray, params: HierParams) -> Dict:
    """
    Build and return a simple hierarchy:
    {
      "meta": {...},
      "parents": [
        {"parent_id": 0, "child_cluster_ids": [0, 1, 2]}
      ]
    }
    """
    # 1) Centroids
    centroids, cluster_ids = compute_cluster_centroids(df, embeddings, params)

    # 2) Parent grouping
    parent_labels = group_clusters(centroids, params)

    # 3) Build parent → children mapping
    parent_to_children: Dict[int, List[int]] = {}
    for idx, pid in enumerate(parent_labels):
        parent_to_children.setdefault(int(pid), []).append(cluster_ids[idx])

    parents_out = []
    for pid in sorted(parent_to_children.keys()):
        parents_out.append({
            "parent_id": pid,
            "child_cluster_ids": sorted(parent_to_children[pid]),
            "parent_summary": "",   # LLM fields not used in demo
            "parent_name": ""
        })

    result = {
        "meta": {
            "n_clusters": len(cluster_ids),
            "n_parents": len(parents_out),
            "params": asdict(params)
        },
        "parents": parents_out
    }

    # 4) Save artifact (optional)
    if params.save_dir:
        outdir = Path(params.save_dir)
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / "hierarchy.json").write_text(json.dumps(result, indent=2))

    return result
