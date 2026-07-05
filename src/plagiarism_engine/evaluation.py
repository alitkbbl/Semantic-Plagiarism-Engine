"""
evaluation.py
=============

Evaluation metrics and pipeline runners used to compare the two detection
backends (MinHash+LSH vs. TF-IDF weighted SimHash) on a labeled pair
dataset, per the "Evaluate on a labeled pair dataset" CLI requirement.

Precision / recall / F1 are implemented from scratch (a two-line
computation) rather than pulled in from scikit-learn, keeping the
dependency footprint to just NumPy + Pandas as specified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd

from . import preprocessing
from .lsh import LSHIndex
from .minhash import MinHasher, estimate_jaccard
from .simhash import TfidfSimHasher, hamming_similarity


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def confusion_counts(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, int]:
    """Compute TP / FP / FN / TN for binary labels (1 = duplicate)."""
    tp = fp = fn = tn = 0
    for t, p in zip(y_true, y_pred):
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def precision_recall_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    """Compute precision, recall and F1 for binary duplicate predictions.

    Returns 0.0 for any metric whose denominator would be zero, rather
    than raising, so evaluation never crashes on a degenerate (e.g.
    all-negative-prediction) run.
    """
    counts = confusion_counts(y_true, y_pred)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, **counts}


def timed(fn: Callable, *args, **kwargs) -> Tuple[object, float]:
    """Run ``fn`` and return (result, elapsed_seconds)."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def sweep_thresholds(
    y_true: Sequence[int], scores: Sequence[float], thresholds: Sequence[float] | None = None,
) -> List[Dict]:
    """Compute precision/recall/F1 at each of a grid of decision thresholds
    applied to a fixed set of similarity ``scores``.

    This computes the underlying similarity score for each pair exactly
    once; sweeping is then just cheap re-thresholding, so it costs almost
    nothing extra on top of a single fixed-threshold evaluation (the
    expensive part -- building MinHash signatures / SimHash fingerprints
    for every pair -- is not repeated per threshold).

    Returns one row per threshold: ``{"threshold", "precision", "recall",
    "f1", "tp", "fp", "fn", "tn"}``, in ascending threshold order.
    """
    if thresholds is None:
        thresholds = [round(i / 100, 2) for i in range(0, 101, 2)]
    rows = []
    for t in thresholds:
        preds = [1 if s >= t else 0 for s in scores]
        metrics = precision_recall_f1(y_true, preds)
        rows.append({"threshold": t, **metrics})
    return rows


def best_threshold_row(sweep_rows: List[Dict]) -> Dict:
    """Pick the sweep row with the highest F1 score.

    Ties are broken in favor of the *first* (i.e. lowest) threshold reached
    while scanning in ascending order, matching the convention used
    throughout this project's threshold-selection experiments (see
    docs/project_spec.pdf, "Parameter Selection").
    """
    best = None
    for row in sweep_rows:
        if best is None or row["f1"] > best["f1"]:
            best = row
    return best


# --------------------------------------------------------------------------- #
# Pipeline runners
# --------------------------------------------------------------------------- #

@dataclass
class PipelineResult:
    method: str
    predictions: List[int]
    elapsed_seconds: float
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class SimilarityScores:
    """Raw per-pair similarity scores for one method, computed once and
    reusable both for a fixed-threshold evaluation and for a threshold
    sweep (see :func:`sweep_thresholds`)."""
    method: str
    scores: List[float]
    elapsed_seconds: float
    extra: Dict[str, float] = field(default_factory=dict)


def compute_minhash_similarities(
    pairs_df: pd.DataFrame,
    text_col_a: str,
    text_col_b: str,
    shingle_size: int = 3,
    num_perm: int = 128,
    num_bands: int = 32,
    compute_lsh_candidates: bool = True,
) -> SimilarityScores:
    """Compute the MinHash-estimated Jaccard similarity for every pair,
    without applying any decision threshold. ``compute_lsh_candidates``
    additionally records, per pair, whether the two documents would have
    landed in the same LSH bucket (a diagnostic separate from the
    similarity score itself, reported as ``lsh_candidate_rate``)."""
    hasher = MinHasher(num_perm=num_perm, seed=42)
    scores: List[float] = []
    lsh_candidate_flags: List[int] = []

    start = time.perf_counter()
    for text_a, text_b in zip(pairs_df[text_col_a], pairs_df[text_col_b]):
        doc_a = preprocessing.preprocess_document(text_a, shingle_size=shingle_size)
        doc_b = preprocessing.preprocess_document(text_b, shingle_size=shingle_size)

        sig_a = hasher.signature(doc_a.shingles)
        sig_b = hasher.signature(doc_b.shingles)
        scores.append(estimate_jaccard(sig_a, sig_b))

        if compute_lsh_candidates:
            index = LSHIndex(num_perm=num_perm, num_bands=num_bands)
            index.insert("a", sig_a)
            index.insert("b", sig_b)
            lsh_candidate_flags.append(1 if "b" in index.query_candidates("a") else 0)
    elapsed = time.perf_counter() - start

    extra = {}
    if compute_lsh_candidates and lsh_candidate_flags:
        extra["lsh_candidate_rate"] = sum(lsh_candidate_flags) / len(lsh_candidate_flags)

    return SimilarityScores(method="minhash_lsh", scores=scores, elapsed_seconds=elapsed, extra=extra)


def compute_simhash_similarities(
    pairs_df: pd.DataFrame,
    text_col_a: str,
    text_col_b: str,
    hash_bits: int = 64,
) -> SimilarityScores:
    """Compute the TF-IDF weighted SimHash Hamming similarity for every
    pair, without applying any decision threshold."""
    scores: List[float] = []

    start = time.perf_counter()
    for text_a, text_b in zip(pairs_df[text_col_a], pairs_df[text_col_b]):
        tokens_a = preprocessing.remove_stopwords(preprocessing.tokenize(text_a))
        tokens_b = preprocessing.remove_stopwords(preprocessing.tokenize(text_b))

        hasher = TfidfSimHasher(hash_bits=hash_bits)
        hasher.fit([tokens_a, tokens_b])

        fp_a = hasher.fingerprint(tokens_a)
        fp_b = hasher.fingerprint(tokens_b)
        scores.append(hamming_similarity(fp_a, fp_b, bits=hash_bits))
    elapsed = time.perf_counter() - start

    return SimilarityScores(method="simhash", scores=scores, elapsed_seconds=elapsed)


def run_minhash_lsh_pipeline(
    pairs_df: pd.DataFrame,
    text_col_a: str,
    text_col_b: str,
    shingle_size: int = 3,
    num_perm: int = 128,
    num_bands: int = 32,
    similarity_threshold: float = 0.5,
) -> PipelineResult:
    """Run the Shingling + MinHash + LSH pipeline over a pair dataset and
    threshold the resulting similarity scores at ``similarity_threshold``.

    A thin wrapper around :func:`compute_minhash_similarities`; kept as a
    separate function (rather than inlining the threshold step everywhere)
    for backward compatibility with existing callers/tests.
    """
    result = compute_minhash_similarities(
        pairs_df, text_col_a, text_col_b,
        shingle_size=shingle_size, num_perm=num_perm, num_bands=num_bands,
    )
    predictions = [1 if s >= similarity_threshold else 0 for s in result.scores]
    return PipelineResult(
        method="minhash_lsh", predictions=predictions,
        elapsed_seconds=result.elapsed_seconds, extra=result.extra,
    )


def run_simhash_pipeline(
    pairs_df: pd.DataFrame,
    text_col_a: str,
    text_col_b: str,
    hash_bits: int = 64,
    hamming_similarity_threshold: float = 0.85,
) -> PipelineResult:
    """Run the TF-IDF weighted SimHash pipeline over a pair dataset and
    threshold the resulting similarity scores at
    ``hamming_similarity_threshold``.

    IDF statistics are fit per-pair over the two documents in that pair
    (matching the "compare two documents" use case) rather than over the
    whole dataset at once, so this function has no cross-row leakage and
    scales identically to the MinHash pipeline above. See the technical
    report for a discussion of this design choice and its trade-offs.

    A thin wrapper around :func:`compute_simhash_similarities`; kept as a
    separate function for backward compatibility with existing
    callers/tests.
    """
    result = compute_simhash_similarities(pairs_df, text_col_a, text_col_b, hash_bits=hash_bits)
    predictions = [1 if s >= hamming_similarity_threshold else 0 for s in result.scores]
    return PipelineResult(method="simhash", predictions=predictions, elapsed_seconds=result.elapsed_seconds)


def evaluate_pairs(
    pairs_df: pd.DataFrame,
    text_col_a: str,
    text_col_b: str,
    label_col: str,
    shingle_size: int = 3,
    num_perm: int = 128,
    num_bands: int = 32,
    minhash_threshold: float = 0.5,
    hash_bits: int = 64,
    simhash_threshold: float = 0.85,
    auto_threshold: bool = False,
    threshold_grid: Sequence[float] | None = None,
) -> Tuple[List[Dict], Dict[str, List[Dict]]]:
    """Run both pipelines over ``pairs_df`` and return one metrics row per
    method, ready to be written to ``outputs/metrics.csv``.

    If ``auto_threshold`` is True, ``minhash_threshold`` / ``simhash_threshold``
    are ignored; instead, each method's similarity scores are computed once
    and then swept across ``threshold_grid`` (default: 0.00 to 1.00 in steps
    of 0.02) to find the threshold that maximizes F1 on this dataset (see
    :func:`sweep_thresholds`). The chosen threshold is recorded in the
    returned row as ``"threshold_used"`` so results stay reproducible.

    Returns ``(rows, sweep_curves)`` where ``sweep_curves`` maps method name
    to its full list of ``{threshold, precision, recall, f1, ...}`` rows
    (empty dict if ``auto_threshold`` is False) -- useful for writing a
    diagnostic sweep CSV or plotting a precision/recall-vs-threshold curve.
    """
    y_true = pairs_df[label_col].astype(int).tolist()

    minhash_scores = compute_minhash_similarities(
        pairs_df, text_col_a, text_col_b,
        shingle_size=shingle_size, num_perm=num_perm, num_bands=num_bands,
    )
    simhash_scores = compute_simhash_similarities(
        pairs_df, text_col_a, text_col_b, hash_bits=hash_bits,
    )

    sweep_curves: Dict[str, List[Dict]] = {}
    fixed_thresholds = {"minhash_lsh": minhash_threshold, "simhash": simhash_threshold}
    thresholds_used: Dict[str, float] = {}

    rows = []
    for result in (minhash_scores, simhash_scores):
        if auto_threshold:
            sweep_rows = sweep_thresholds(y_true, result.scores, thresholds=threshold_grid)
            sweep_curves[result.method] = sweep_rows
            chosen = best_threshold_row(sweep_rows)
            threshold_used = chosen["threshold"]
            metrics = chosen
        else:
            threshold_used = fixed_thresholds[result.method]
            predictions = [1 if s >= threshold_used else 0 for s in result.scores]
            metrics = precision_recall_f1(y_true, predictions)
        thresholds_used[result.method] = threshold_used

        row = {
            "method": result.method,
            "num_pairs": len(y_true),
            "threshold_used": threshold_used,
            "precision": round(metrics["precision"], 4),
            "recall": round(metrics["recall"], 4),
            "f1": round(metrics["f1"], 4),
            "true_positive": metrics["tp"],
            "false_positive": metrics["fp"],
            "false_negative": metrics["fn"],
            "true_negative": metrics["tn"],
            "total_time_seconds": round(result.elapsed_seconds, 4),
            "avg_time_per_pair_ms": round(
                1000 * result.elapsed_seconds / len(y_true), 4
            )
            if y_true
            else 0.0,
        }
        row.update({k: round(v, 4) for k, v in result.extra.items()})
        rows.append(row)

    return rows, sweep_curves
