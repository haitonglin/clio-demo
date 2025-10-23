"""
projector.py
============
UMAP-based 2-D projector for visual inspection (paper-style).
Note: This is *purely* a visualization aid; it does not affect clustering.

Public API:
- ProjectorParams                 # config
- compute_umap(embeddings, params) -> np.ndarray (N, 2)
- project_and_merge(df, embeddings, params, x_col="umap_x", y_col="umap_y") -> pd.DataFrame
- plot_projection(df, cluster_col, x_col="umap_x", y_col="umap_y") -> None  (optional)

Usage pattern in notebook:
    from projector import ProjectorParams, project_and_merge, plot_projection
    params = ProjectorParams(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=123)
    df_proj = project_and_merge(clustered_df, X, params)
    plot_projection(df_proj, cluster_col="cluster_id")
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import umap
except ImportError as e:
    raise ImportError(
        "The 'umap-learn' package is required. Install with `pip install umap-learn`."
    ) from e

import matplotlib.pyplot as plt


# =========================
# Configuration / Defaults
# =========================

@dataclass
class ProjectorParams:
    """Configuration for UMAP projector (paper-style defaults)."""
    n_neighbors: int = 15        # small neighborhood preserves local clusters
    min_dist: float = 0.1        # tighter clusters on the plane
    metric: str = "cosine"       # cosine distance for sentence embeddings
    random_state: int = 42       # deterministic demo runs
    n_components: int = 2        # 2D for projector view
    cache_dir: Optional[str] = "artifacts/projector"  # set to None to disable caching
    cache_key: Optional[str] = None  # override to force unique cache runs, else auto-derived


# =========================
# Core functions
# =========================

def _maybe_load_cache(N: int, params: ProjectorParams) -> Optional[np.ndarray]:
    """Load cached UMAP coords if present and compatible."""
    if not params.cache_dir:
        return None
    cache_dir = Path(params.cache_dir)
    x_path = cache_dir / "umap_coords.npy"
    meta_path = cache_dir / "umap_meta.json"
    if not x_path.exists() or not meta_path.exists():
        return None
    try:
        coords = np.load(x_path)
        if coords.shape[0] != N or coords.shape[1] != 2:
            return None
        meta = json.loads(meta_path.read_text())
        # simple compatibility check
        keys = ["n_neighbors", "min_dist", "metric", "random_state", "n_components", "cache_key"]
        if {k: meta.get(k) for k in keys} == {k: getattr(params, k) for k in keys}:
            return coords
    except Exception:
        return None
    return None  # set to coords to enable cache reuse; left off by default for clarity


def _maybe_save_cache(coords: np.ndarray, params: ProjectorParams) -> None:
    """Save UMAP coords + params for reproducible demos."""
    if not params.cache_dir:
        return
    cache_dir = Path(params.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "umap_coords.npy", coords)
    meta = asdict(params)
    (cache_dir / "umap_meta.json").write_text(json.dumps(meta, indent=2))


def compute_umap(embeddings: np.ndarray, params: ProjectorParams) -> np.ndarray:
    """
    Compute 2-D UMAP projection (paper-style). Returns array of shape (N, 2).

    Notes:
    - UMAP is non-deterministic without a seed; we set random_state.
    - Uses cosine metric to match sentence transformer geometry.
    - This function does *not* alter clustering; it's visualization-only.
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a numpy.ndarray of shape (N, d)")
    if embeddings.ndim != 2 or embeddings.shape[0] < 2:
        raise ValueError("embeddings must be 2D and have at least 2 rows")

    # Optional cache (disabled load by default above for simplicity; enable if desired)
    cached = _maybe_load_cache(embeddings.shape[0], params)
    if cached is not None:
        return cached

    reducer = umap.UMAP(
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        metric=params.metric,
        n_components=params.n_components,
        random_state=params.random_state,
        # UMAP uses pynndescent by default, good for large-N
    )
    coords = reducer.fit_transform(embeddings)  # shape (N, 2)

    _maybe_save_cache(coords, params)
    return coords


def project_and_merge(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    params: ProjectorParams,
    x_col: str = "umap_x",
    y_col: str = "umap_y"
) -> pd.DataFrame:
    """
    Compute UMAP and merge (x, y) back into df.
    Returns a *new* dataframe with added columns [x_col, y_col].
    """
    if len(df) != embeddings.shape[0]:
        raise ValueError("len(df) must match embeddings.shape[0]")
    coords = compute_umap(embeddings, params)
    out = df.copy()
    out[x_col] = coords[:, 0]
    out[y_col] = coords[:, 1]
    return out


# =========================
# Optional plotting helper
# =========================

def plot_projection(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    x_col: str = "umap_x",
    y_col: str = "umap_y",
    title: str = "UMAP Projector (paper-style)"
) -> None:
    """
    Simple scatter for quick inspection. For production UI, use your own front-end.
    - No seaborn, one plot, no specific colors (keeps things lightweight).
    """
    required = {cluster_col, x_col, y_col}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Missing columns for plotting: {missing}")

    plt.figure(figsize=(7, 6))
    # draw each cluster separately to get discrete legend entries
    for cid, group in df.groupby(cluster_col):
        plt.scatter(group[x_col], group[y_col], s=14, alpha=0.75, label=f"C{int(cid)}")
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.legend(title=cluster_col, loc="best", fontsize=9)
    plt.tight_layout()
    plt.show()
