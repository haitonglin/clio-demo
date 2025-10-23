"""
cluster_naming.py
=================
Paper-aligned cluster naming utilities (Clio-style), simplified for a demo:

Public functions:
- sample_cluster_examples(df, cluster_id, embeddings, params=None)
- build_cluster_prompt(answers, contrastive_answers, criteria=DEFAULT_CRITERIA)
- generate_cluster_name(df, cluster_id, embeddings, complete_fn, params=None, criteria=None)
- name_all_clusters(df, embeddings, complete_fn, params=None, criteria=None)

Key simplification:
- Instead of an adapter, pass a plain callable: complete_fn(prompt: str) -> str
  Example: complete_fn = ClaudeClient(...).complete

Notes:
- EXACT procedure: ≤50 in-cluster + 50 nearest-out-of-cluster; temp=1 (handled in your ClaudeClient if applicable)
- Embeddings are passed as a separate matrix (np.ndarray) to match your pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import pandas as pd
import random
import re
from pathlib import Path
from collections import Counter


# =========================
# Configuration / Defaults
# =========================

@dataclass
class NamingParams:
    """Runtime parameters for naming (safe defaults mirror the paper)."""
    in_cluster_n: int = 50
    out_cluster_n: int = 50
    text_col: str = "facet_summary"
    cluster_col: str = "cluster_id"
    random_state: int = 42
    save_dir: Optional[str] = "artifacts/naming"  # set to None to skip saving

DEFAULT_CRITERIA = (
    "For the Request facet: the cluster name must be a sentence in the imperative "
    "(e.g., 'Brainstorm ideas for a birthday party'). At most ten words."
)


# =========================
# Internal Helper Functions
# =========================

def _rng(seed: int) -> random.Random:
    r = random.Random()
    r.seed(seed)
    return r

def _compute_centroid(vecs: np.ndarray) -> np.ndarray:
    """Centroid = mean vector."""
    return vecs.mean(axis=0)

def _nearest_out_of_cluster(
    embeddings: np.ndarray,   # (N, d)
    in_mask: np.ndarray,      # bool mask
    centroid: np.ndarray,     # (d,)
    k: int
) -> List[int]:
    dists = np.linalg.norm(embeddings - centroid[None, :], axis=1)
    candidates = np.where(~in_mask)[0]
    if candidates.size == 0:
        return []
    order = np.argsort(dists[candidates])
    return candidates[order[: min(k, candidates.size)]].tolist()

def _xml_join(lines: List[str]) -> str:
    return "\n".join(s.strip() for s in lines if isinstance(s, str) and s.strip())

def _parse_xml_response(text: str) -> Tuple[str, str]:
    s = re.search(r"<summary>\s*(.*?)\s*</summary>", text, flags=re.S | re.I)
    n = re.search(r"<name>\s*(.*?)\s*</name>", text, flags=re.S | re.I)
    return (s.group(1).strip() if s else "", n.group(1).strip() if n else "")

# Very small fallback if LLM call errors: generate a presentable name/summary from tokens.
_STOPWORDS = set("""
the a an and or of to for in on with from by as about this that those these is are was were be been being
i you he she it we they my your his her their our me us them do did done does doing have has had having
""".split())

def _fallback_name_summary(texts: List[str]) -> Tuple[str, str]:
    tokens = []
    for t in texts:
        tokens += re.findall(r"[A-Za-z][A-Za-z\-']{2,}", t.lower())
    tokens = [t for t in tokens if t not in _STOPWORDS]
    common = [w for w, _c in Counter(tokens).most_common(6)]
    # crude imperative-ish name
    name = " ".join(["Summarize"] + [w.capitalize() for w in common[:4]])[:80] or "Summarize Cluster"
    # crude past-tense two-sentence summary
    joined = " ".join(texts)[:300]
    summary = (
        "Participants requested or discussed topics matching a coherent theme. "
        "This cluster differed from nearby groups by emphasizing: " + ", ".join(common[:6])
    )
    return summary, name


# ==================================
# 1) sample_cluster_examples (API)
# ==================================

def sample_cluster_examples(
    df: pd.DataFrame,
    cluster_id: int,
    embeddings: np.ndarray,
    params: Optional[NamingParams] = None
) -> Dict[str, List]:
    """
    Paper procedure:
      - In-cluster: random sample up to 50
      - Out-of-cluster: 50 nearest to the centroid but outside the cluster
    """
    p = params or NamingParams()
    text_col, ccol = p.text_col, p.cluster_col

    if not {text_col, ccol}.issubset(df.columns):
        raise ValueError(f"df must contain columns: {text_col}, {ccol}")
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a numpy.ndarray of shape (N, d)")
    if len(df) != embeddings.shape[0]:
        raise ValueError("len(df) must match embeddings.shape[0]")

    in_mask = (df[ccol].values == cluster_id)
    in_idx = np.where(in_mask)[0].tolist()

    r = _rng(p.random_state + int(cluster_id))
    in_idx_sampled = r.sample(in_idx, min(len(in_idx), p.in_cluster_n))

    in_vecs = embeddings[in_mask]
    if in_vecs.shape[0] == 0:
        return {"in_indices": [], "out_indices": [], "in_texts": [], "out_texts": []}

    centroid = _compute_centroid(in_vecs)
    out_idx_near = _nearest_out_of_cluster(embeddings, in_mask, centroid, p.out_cluster_n)

    in_texts = df.iloc[in_idx_sampled][text_col].astype(str).tolist()
    out_texts = df.iloc[out_idx_near][text_col].astype(str).tolist()

    return {
        "in_indices": in_idx_sampled,
        "out_indices": out_idx_near,
        "in_texts": in_texts,
        "out_texts": out_texts
    }


# =================================
# 2) build_cluster_prompt (API)
# =================================

def build_cluster_prompt(
    answers: List[str],
    contrastive_answers: List[str],
    criteria: str = DEFAULT_CRITERIA
) -> str:
    """
    Exact Clio-style prompt. Output must be:
      <summary>...</summary>
      <name>...</name>
    """
    return (
        "You are tasked with summarizing a group of related statements into a short, precise, and accurate description and name. "
        "Your goal is to create a concise summary that captures the essence of these statements and distinguishes it from other similar groups of statements.\n\n"
        "Summarize all the statements into a clear, precise, two-sentence description in the past tense. "
        "Your summary should be specific to this group and distinguish it from the contrastive answers of the other groups.\n\n"
        "After creating the summary, generate a short name for the group of statements. This name should be at most ten words long (perhaps less) "
        "and be specific but also reflective of most of the statements (rather than reflecting only one or two). "
        "The name should distinguish this group from the contrastive examples. For instance, \"Write fantasy sexual roleplay with octopi and monsters\", "
        "\"Generate blog spam for gambling websites\", or \"Assist with high school math homework\" would be better and more actionable than general terms like "
        "\"Write erotic content\" or \"Help with homework\". Be as descriptive as possible and assume neither good nor bad faith. "
        "Do not hesitate to identify and describe socially harmful or sensitive topics specifically; specificity is necessary for monitoring.\n\n"
        "Present your output in the following format:\n"
        "<summary> [Insert your two-sentence summary here] </summary>\n"
        "<name> [Insert your generated short name here] </name>\n\n"
        "The names you propose must follow these requirements:\n"
        f"<criteria>\n{criteria}\n</criteria>\n\n"
        "Below are the related statements:\n"
        "<answers>\n"
        f"{_xml_join(answers)}\n"
        "</answers>\n\n"
        "For context, here are statements from nearby groups that are NOT part of the group you are summarizing:\n"
        "<contrastive_answers>\n"
        f"{_xml_join(contrastive_answers)}\n"
        "</contrastive_answers>\n\n"
        "Do not elaborate beyond what you say in the tags. Remember to analyze both the statements and the contrastive statements carefully "
        "to ensure your summary and name accurately represent the specific group while distinguishing it from others.\n\n"
        "Assistant: Sure, I will provide a clear, precise, and accurate summary and name for this cluster. I will be descriptive and assume neither good nor bad faith. "
        "Here is the summary, which I will follow with the name: <summary>"
    )


# ===================================
# 3) generate_cluster_name (API)
# ===================================

def generate_cluster_name(
    df: pd.DataFrame,
    cluster_id: int,
    embeddings: np.ndarray,
    complete_fn: Callable[[str], str],   # ← plain function: prompt -> text
    params: Optional[NamingParams] = None,
    criteria: Optional[str] = None
) -> Dict:
    """
    Build the prompt for a specific cluster (paper procedure) and call the LLM via complete_fn.
    """
    p = params or NamingParams()
    crit = criteria if criteria is not None else DEFAULT_CRITERIA

    picked = sample_cluster_examples(df, cluster_id, embeddings, params=p)
    prompt = build_cluster_prompt(picked["in_texts"], picked["out_texts"], criteria=crit)

    # Call LLM; if it fails, use a safe fallback so demos still render.
    try:
        raw = complete_fn(prompt)
        summary, name = _parse_xml_response(raw)
        if not summary or not name:
            # model responded but missing tags → fallback
            summary, name = _fallback_name_summary(picked["in_texts"])
    except Exception:
        summary, name = _fallback_name_summary(picked["in_texts"])

    prompt_path = output_path = None
    if p.save_dir:
        Path(p.save_dir).mkdir(parents=True, exist_ok=True)
        prompt_path = str(Path(p.save_dir) / f"cluster_{cluster_id:03d}_prompt.txt")
        output_path = str(Path(p.save_dir) / f"cluster_{cluster_id:03d}_output.txt")
        Path(prompt_path).write_text(prompt)
        # Write whatever we got (either raw model output or synthesized XML)
        try:
            Path(output_path).write_text(raw)  # may not exist if exception
        except Exception:
            Path(output_path).write_text(f"<summary>{summary}</summary>\n<name>{name}</name>")

    return {
        "cluster_id": int(cluster_id),
        "summary": summary,
        "name": name,
        "in_sample_size": len(picked["in_texts"]),
        "out_sample_size": len(picked["out_texts"]),
        "prompt_path": prompt_path,
        "output_path": output_path
    }


# ==============================
# 4) name_all_clusters (API)
# ==============================

def name_all_clusters(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    complete_fn: Callable[[str], str],
    params: Optional[NamingParams] = None,
    criteria: Optional[str] = None
) -> pd.DataFrame:
    """
    Loop through unique cluster_ids and return:
        | cluster_id | summary | name |
    """
    p = params or NamingParams()
    ccol = p.cluster_col
    if ccol not in df.columns:
        raise ValueError(f"df must contain column '{ccol}'")
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a numpy.ndarray of shape (N, d)")
    if len(df) != embeddings.shape[0]:
        raise ValueError("len(df) must match embeddings.shape[0]")

    records = []
    for cid in sorted(df[ccol].dropna().unique().tolist()):
        out = generate_cluster_name(df, int(cid), embeddings, complete_fn, params=p, criteria=criteria)
        records.append(out)

    names_df = pd.DataFrame(records)[["cluster_id", "summary", "name"]]

    if p.save_dir:
        out_dir = Path(p.save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        names_df.to_csv(out_dir / "cluster_names.csv", index=False)
        names_df.to_json(out_dir / "cluster_names.jsonl", orient="records", lines=True)

    return names_df

