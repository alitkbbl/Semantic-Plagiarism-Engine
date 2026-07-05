"""
cli.py
======

Command line interface for the Semantic Duplicate & Near-Plagiarism
Detection Engine.

Provides three commands, per the project specification:

    compare   Compare two documents directly.
    corpus    Search for similar documents inside a folder.
    pairs     Evaluate both backends on a labeled pair dataset.

Run ``python -m plagiarism_engine.cli --help`` (or, once installed,
``plagiarism-engine --help``) for full usage.
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import click

from . import dataset, preprocessing
from .evaluation import evaluate_pairs
from .lsh import LSHIndex, s_curve_threshold
from .minhash import MinHasher, estimate_jaccard
from .simhash import TfidfSimHasher, hamming_distance, hamming_similarity


def _exact_jaccard(shingles_a: set, shingles_b: set) -> float:
    """Exact Jaccard similarity of two shingle sets.

    By convention J(empty, empty) = 1.0 (two empty documents are
    identical), and J(empty, non-empty) = 0.0.
    """
    if not shingles_a and not shingles_b:
        return 1.0
    union = shingles_a | shingles_b
    if not union:
        return 0.0
    return len(shingles_a & shingles_b) / len(union)


@click.group()
@click.version_option(package_name="plagiarism-engine")
def cli() -> None:
    """Semantic Duplicate & Near-Plagiarism Detection Engine."""


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #

@cli.command()
@click.option("--file-a", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to the first document.")
@click.option("--file-b", required=True, type=click.Path(exists=True, dir_okay=False), help="Path to the second document.")
@click.option("--shingle-size", default=3, show_default=True, help="Word shingle size (k).")
@click.option("--num-perm", default=128, show_default=True, help="MinHash signature length.")
@click.option("--num-bands", default=32, show_default=True, help="Number of LSH bands (must divide num-perm).")
@click.option("--hash-bits", default=64, show_default=True, help="SimHash fingerprint width in bits.")
@click.option("--output", default=None, type=click.Path(dir_okay=False), help="Optional path to write a JSON result.")
def compare(file_a, file_b, shingle_size, num_perm, num_bands, hash_bits, output):
    """Compare two documents with both detection backends."""
    text_a = Path(file_a).read_text(encoding="utf-8", errors="replace")
    text_b = Path(file_b).read_text(encoding="utf-8", errors="replace")

    doc_a = preprocessing.preprocess_document(text_a, shingle_size=shingle_size)
    doc_b = preprocessing.preprocess_document(text_b, shingle_size=shingle_size)

    # --- Shingling + MinHash + LSH -----------------------------------
    t0 = time.perf_counter()
    exact_jaccard = _exact_jaccard(doc_a.shingles, doc_b.shingles)
    t_exact = time.perf_counter() - t0

    hasher = MinHasher(num_perm=num_perm)
    t0 = time.perf_counter()
    sig_a = hasher.signature(doc_a.shingles)
    sig_b = hasher.signature(doc_b.shingles)
    minhash_estimate = estimate_jaccard(sig_a, sig_b)
    t_minhash = time.perf_counter() - t0

    index = LSHIndex(num_perm=num_perm, num_bands=num_bands)
    index.insert("A", sig_a)
    index.insert("B", sig_b)
    is_lsh_candidate = "B" in index.query_candidates("A")
    lsh_threshold = s_curve_threshold(num_bands, num_perm // num_bands)

    # --- TF-IDF weighted SimHash --------------------------------------
    simhasher = TfidfSimHasher(hash_bits=hash_bits)
    simhasher.fit([doc_a.tokens, doc_b.tokens])
    t0 = time.perf_counter()
    fp_a = simhasher.fingerprint(doc_a.tokens)
    fp_b = simhasher.fingerprint(doc_b.tokens)
    t_simhash = time.perf_counter() - t0
    hdist = hamming_distance(fp_a, fp_b)
    hsim = hamming_similarity(fp_a, fp_b, bits=hash_bits)

    result = {
        "file_a": str(file_a),
        "file_b": str(file_b),
        "preprocessing": {
            "shingle_size": shingle_size,
            "tokens_a": len(doc_a.tokens),
            "tokens_b": len(doc_b.tokens),
            "shingles_a": len(doc_a.shingles),
            "shingles_b": len(doc_b.shingles),
        },
        "shingling_minhash_lsh": {
            "exact_jaccard_similarity": round(exact_jaccard, 6),
            "minhash_estimated_similarity": round(minhash_estimate, 6),
            "num_perm": num_perm,
            "num_bands": num_bands,
            "rows_per_band": num_perm // num_bands,
            "lsh_candidate_pair": is_lsh_candidate,
            "lsh_approx_detection_threshold": round(lsh_threshold, 4),
            "exact_jaccard_time_seconds": round(t_exact, 6),
            "minhash_time_seconds": round(t_minhash, 6),
        },
        "tfidf_simhash": {
            "hash_bits": hash_bits,
            "fingerprint_a_hex": format(fp_a, f"0{hash_bits // 4}x"),
            "fingerprint_b_hex": format(fp_b, f"0{hash_bits // 4}x"),
            "hamming_distance": hdist,
            "hamming_similarity": round(hsim, 6),
            "simhash_time_seconds": round(t_simhash, 6),
        },
    }

    click.echo(click.style(f"File A: {file_a}", fg="cyan"))
    click.echo(click.style(f"File B: {file_b}", fg="cyan"))
    click.echo("")
    click.echo(f"Exact Jaccard similarity      : {exact_jaccard:.4f}")
    click.echo(f"MinHash estimated similarity  : {minhash_estimate:.4f}  (num_perm={num_perm})")
    click.echo(f"LSH candidate pair            : {is_lsh_candidate}  (bands={num_bands})")
    click.echo(f"SimHash Hamming distance      : {hdist} / {hash_bits} bits")
    click.echo(f"SimHash Hamming similarity    : {hsim:.4f}")

    if output:
        dataset.save_json(result, Path(output))
        click.echo(click.style(f"\nWrote JSON result to {output}", fg="green"))


# --------------------------------------------------------------------------- #
# corpus
# --------------------------------------------------------------------------- #

@cli.command()
@click.option("--data", required=True, type=click.Path(exists=True, file_okay=False), help="Folder of *.txt documents.")
@click.option("--threshold", default=0.25, show_default=True, help="Minimum exact Jaccard similarity to report a pair as a duplicate.")
@click.option("--shingle-size", default=3, show_default=True, help="Word shingle size (k).")
@click.option("--num-perm", default=128, show_default=True, help="MinHash signature length.")
@click.option("--num-bands", default=32, show_default=True, help="Number of LSH bands (must divide num-perm).")
@click.option("--output", required=True, type=click.Path(dir_okay=False), help="Path to write candidates CSV.")
def corpus(data, threshold, shingle_size, num_perm, num_bands, output):
    """Search a folder of documents for near-duplicate / similar pairs.

    Uses Shingling + MinHash + LSH to cut down the number of expensive
    exact-Jaccard comparisons from the full O(n^2) pair set to just the
    LSH candidate pairs, then verifies each candidate with exact Jaccard
    and keeps the ones at or above ``--threshold``.
    """
    documents = dataset.load_corpus(Path(data))
    doc_ids = list(documents.keys())
    n = len(doc_ids)

    if n < 2:
        click.echo(click.style("Need at least 2 documents to compare.", fg="red"))
        dataset.save_rows_csv([], Path(output), fieldnames=[
            "doc_a", "doc_b", "exact_jaccard_similarity", "minhash_estimated_similarity",
        ])
        return

    hasher = MinHasher(num_perm=num_perm)
    preprocessed = {}
    signatures = {}

    t0 = time.perf_counter()
    index = LSHIndex(num_perm=num_perm, num_bands=num_bands)
    for doc_id in doc_ids:
        pdoc = preprocessing.preprocess_document(documents[doc_id], shingle_size=shingle_size)
        sig = hasher.signature(pdoc.shingles)
        preprocessed[doc_id] = pdoc
        signatures[doc_id] = sig
        index.insert(doc_id, sig)
    indexing_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    candidate_pairs = index.candidate_pairs()
    lsh_time = time.perf_counter() - t0

    rows = []
    t0 = time.perf_counter()
    for doc_a, doc_b in sorted(candidate_pairs):
        sim = _exact_jaccard(preprocessed[doc_a].shingles, preprocessed[doc_b].shingles)
        if sim >= threshold:
            minhash_sim = estimate_jaccard(signatures[doc_a], signatures[doc_b])
            rows.append({
                "doc_a": doc_a,
                "doc_b": doc_b,
                "exact_jaccard_similarity": round(sim, 6),
                "minhash_estimated_similarity": round(minhash_sim, 6),
            })
    verification_time = time.perf_counter() - t0

    rows.sort(key=lambda r: r["exact_jaccard_similarity"], reverse=True)
    dataset.save_rows_csv(
        rows, Path(output),
        fieldnames=["doc_a", "doc_b", "exact_jaccard_similarity", "minhash_estimated_similarity"],
    )

    total_possible_pairs = n * (n - 1) // 2
    num_candidates = len(candidate_pairs)
    reduction_pct = (
        100.0 * (1 - num_candidates / total_possible_pairs) if total_possible_pairs else 0.0
    )

    click.echo(click.style(f"Documents scanned            : {n}", fg="cyan"))
    click.echo(f"Total possible pairs (O(n^2))  : {total_possible_pairs}")
    click.echo(f"LSH candidate pairs             : {num_candidates}")
    click.echo(f"Comparisons avoided by LSH       : {reduction_pct:.2f}%")
    click.echo(f"Pairs above threshold ({threshold}) : {len(rows)}")
    click.echo(f"Indexing time                   : {indexing_time:.4f}s")
    click.echo(f"LSH candidate lookup time        : {lsh_time:.4f}s")
    click.echo(f"Exact verification time          : {verification_time:.4f}s")
    click.echo(click.style(f"\nWrote {len(rows)} rows to {output}", fg="green"))


# --------------------------------------------------------------------------- #
# pairs
# --------------------------------------------------------------------------- #

@cli.command()
@click.option("--pairs", "pairs_path", required=True, type=click.Path(exists=True, dir_okay=False), help="CSV file of labeled document pairs.")
@click.option("--text-col-a", required=True, help="Column name of the first document's text.")
@click.option("--text-col-b", required=True, help="Column name of the second document's text.")
@click.option("--label-col", required=True, help="Column name of the binary duplicate label (1/0).")
@click.option("--limit", default=None, type=int, help="Only evaluate the first N rows.")
@click.option("--shingle-size", default=3, show_default=True, help="Word shingle size (k) for MinHash.")
@click.option("--num-perm", default=128, show_default=True, help="MinHash signature length.")
@click.option("--num-bands", default=32, show_default=True, help="Number of LSH bands (must divide num-perm).")
@click.option("--minhash-threshold", default=0.5, show_default=True, help="MinHash similarity threshold for a 'duplicate' prediction (ignored if --sweep is given).")
@click.option("--hash-bits", default=64, show_default=True, help="SimHash fingerprint width in bits.")
@click.option("--simhash-threshold", default=0.85, show_default=True, help="SimHash Hamming-similarity threshold for a 'duplicate' prediction (ignored if --sweep is given).")
@click.option("--sweep", is_flag=True, default=False, help="Ignore the fixed thresholds above; instead sweep a threshold grid per method and report the F1-maximizing threshold. Costs almost nothing extra: similarity scores are computed once and re-thresholded cheaply.")
@click.option("--sweep-output", default=None, type=click.Path(dir_okay=False), help="Optional path to write the full threshold/precision/recall/F1 curve for both methods (requires --sweep).")
@click.option("--output", required=True, type=click.Path(dir_okay=False), help="Path to write metrics CSV.")
def pairs(
    pairs_path, text_col_a, text_col_b, label_col, limit,
    shingle_size, num_perm, num_bands, minhash_threshold,
    hash_bits, simhash_threshold, sweep, sweep_output, output,
):
    """Evaluate both backends on a labeled pair dataset (e.g. Quora
    Question Pairs, Stack Exchange Duplicates, or a PAN-PC-11-derived
    pair list) and report precision / recall / F1 / timing.

    Similarity-score distributions vary a lot by document length and by
    how heavily a dataset's positive pairs are paraphrased/obfuscated, so
    a threshold tuned for one dataset (e.g. short questions) can perform
    very poorly on another (e.g. literary text with obfuscated plagiarism)
    -- very high precision but collapsed recall is the tell-tale sign of
    a too-strict threshold, not necessarily a broken similarity measure.
    Use --sweep to find a dataset-appropriate threshold automatically
    instead of guessing.
    """
    pairs_df = dataset.load_pairs_csv(
        Path(pairs_path), text_col_a, text_col_b, label_col=label_col, limit=limit,
    )
    if pairs_df.empty:
        click.echo(click.style("No valid rows found in the pairs file.", fg="red"))
        return

    rows, sweep_curves = evaluate_pairs(
        pairs_df, text_col_a, text_col_b, label_col,
        shingle_size=shingle_size, num_perm=num_perm, num_bands=num_bands,
        minhash_threshold=minhash_threshold,
        hash_bits=hash_bits, simhash_threshold=simhash_threshold,
        auto_threshold=sweep,
    )

    fieldnames = list(rows[0].keys()) if rows else []
    dataset.save_rows_csv(rows, Path(output), fieldnames=fieldnames)

    if sweep_output and sweep_curves:
        sweep_rows = []
        for method, curve in sweep_curves.items():
            for point in curve:
                sweep_rows.append({"method": method, **point})
        sweep_fieldnames = list(sweep_rows[0].keys()) if sweep_rows else []
        dataset.save_rows_csv(sweep_rows, Path(sweep_output), fieldnames=sweep_fieldnames)

    click.echo(click.style(f"Evaluated {len(pairs_df)} labeled pairs\n", fg="cyan"))
    if sweep:
        click.echo(click.style("(--sweep enabled: thresholds below were chosen to maximize F1, "
                                "not read from --minhash-threshold/--simhash-threshold)\n", fg="yellow"))
    header = f"{'method':<14}{'threshold':>10}{'precision':>10}{'recall':>10}{'f1':>10}{'time(s)':>10}"
    click.echo(header)
    click.echo("-" * len(header))
    for row in rows:
        click.echo(
            f"{row['method']:<14}{row['threshold_used']:>10.4f}{row['precision']:>10.4f}{row['recall']:>10.4f}"
            f"{row['f1']:>10.4f}{row['total_time_seconds']:>10.4f}"
        )
    click.echo(click.style(f"\nWrote metrics to {output}", fg="green"))
    if sweep_output and sweep_curves:
        click.echo(click.style(f"Wrote full threshold sweep to {sweep_output}", fg="green"))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
