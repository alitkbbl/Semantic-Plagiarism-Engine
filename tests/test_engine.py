"""
test_engine.py
===============

Unit and light integration tests for the plagiarism_engine package.

Run with:

    pytest
    # or, for verbose output:
    pytest -v
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from plagiarism_engine import preprocessing
from plagiarism_engine.dataset import load_corpus, load_pairs_csv, save_rows_csv
from plagiarism_engine.evaluation import (
    best_threshold_row,
    confusion_counts,
    evaluate_pairs,
    precision_recall_f1,
    run_minhash_lsh_pipeline,
    run_simhash_pipeline,
    sweep_thresholds,
)
from plagiarism_engine.lsh import LSHIndex, candidate_probability, s_curve_threshold
from plagiarism_engine.minhash import MinHasher, estimate_jaccard, stable_hash
from plagiarism_engine.simhash import (
    TfidfSimHasher,
    hamming_distance,
    hamming_similarity,
    stable_hash_64,
)


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #

class TestPreprocessing:
    def test_normalize_lowercases_and_strips_punctuation(self):
        text = "Hello, World!!  How ARE you??"
        normalized = preprocessing.normalize_text(text)
        assert normalized == "hello world how are you"

    def test_normalize_collapses_whitespace(self):
        assert preprocessing.normalize_text("a   b\t\tc\n\nd") == "a b c d"

    def test_normalize_handles_empty_and_none(self):
        assert preprocessing.normalize_text("") == ""
        assert preprocessing.normalize_text(None) == ""

    def test_normalize_handles_unusual_characters(self):
        # Emoji, control characters, and accented letters should not crash
        # the pipeline; control chars are stripped, everything else is
        # lower-cased/normalized.
        text = "Caf\u00e9 \U0001F30D so\u0301 \x00\x01 nice"
        normalized = preprocessing.normalize_text(text)
        assert "\x00" not in normalized
        assert "\x01" not in normalized
        assert "caf" in normalized

    def test_persian_character_folding(self):
        # Arabic YEH / KAF should fold to the Persian equivalents so that
        # visually-identical Persian text normalizes identically regardless
        # of which keyboard/input method produced it.
        arabic_form = "\u0643\u062a\u0627\u0628"   # ARABIC KAF + ...
        persian_form = "\u06a9\u062a\u0627\u0628"  # PERSIAN KEHEH + ...
        assert preprocessing.normalize_text(arabic_form) == preprocessing.normalize_text(persian_form)

    def test_tokenize_empty_document(self):
        assert preprocessing.tokenize("") == []
        assert preprocessing.tokenize("   ") == []

    def test_remove_stopwords_english(self):
        tokens = ["the", "quick", "brown", "fox", "is", "fast"]
        filtered = preprocessing.remove_stopwords(tokens)
        assert filtered == ["quick", "brown", "fox", "fast"]

    def test_remove_stopwords_persian(self):
        tokens = ["این", "کتاب", "خوب", "است"]
        filtered = preprocessing.remove_stopwords(tokens)
        assert filtered == ["کتاب", "خوب"]

    def test_build_shingles_normal(self):
        tokens = ["a", "b", "c", "d"]
        shingles = preprocessing.build_shingles(tokens, k=2)
        assert shingles == {"a b", "b c", "c d"}

    def test_build_shingles_empty_document(self):
        assert preprocessing.build_shingles([], k=3) == set()

    def test_build_shingles_short_document(self):
        # Fewer tokens than k: the whole document becomes a single shingle
        # instead of being silently dropped.
        tokens = ["only", "two"]
        shingles = preprocessing.build_shingles(tokens, k=5)
        assert shingles == {"only two"}

    def test_build_shingles_invalid_k(self):
        with pytest.raises(ValueError):
            preprocessing.build_shingles(["a", "b"], k=0)

    def test_preprocess_document_empty_text(self):
        doc = preprocessing.preprocess_document("")
        assert doc.is_empty
        assert doc.tokens == []
        assert doc.shingles == set()

    def test_preprocess_document_full_pipeline(self):
        doc = preprocessing.preprocess_document(
            "The quick brown fox jumps over the lazy dog.", shingle_size=3
        )
        assert "the" not in doc.tokens  # stop word removed
        assert len(doc.shingles) > 0


# --------------------------------------------------------------------------- #
# minhash
# --------------------------------------------------------------------------- #

class TestMinHash:
    def test_stable_hash_deterministic(self):
        assert stable_hash("hello") == stable_hash("hello")
        assert stable_hash("hello") != stable_hash("world")

    def test_signature_length_and_dtype(self):
        hasher = MinHasher(num_perm=64)
        sig = hasher.signature({"a b c", "b c d"})
        assert len(sig) == 64
        assert sig.dtype == np.uint64

    def test_signature_deterministic(self):
        hasher = MinHasher(num_perm=32, seed=1)
        shingles = {"the cat sat", "cat sat on", "sat on the"}
        sig1 = hasher.signature(shingles)
        sig2 = hasher.signature(shingles)
        assert np.array_equal(sig1, sig2)

    def test_identical_sets_have_similarity_one(self):
        hasher = MinHasher(num_perm=128, seed=0)
        shingles = {"a b c", "b c d", "c d e"}
        sig = hasher.signature(shingles)
        assert estimate_jaccard(sig, sig) == 1.0

    def test_empty_sets_have_similarity_one(self):
        hasher = MinHasher(num_perm=64)
        sig_a = hasher.signature(set())
        sig_b = hasher.signature(set())
        assert estimate_jaccard(sig_a, sig_b) == 1.0

    def test_minhash_estimate_close_to_exact_jaccard(self):
        # With enough permutations, the MinHash estimate should land
        # within a reasonable tolerance of the true Jaccard similarity.
        a = {f"shingle_{i}" for i in range(80)}
        b = {f"shingle_{i}" for i in range(40, 120)}
        exact = len(a & b) / len(a | b)

        hasher = MinHasher(num_perm=256, seed=3)
        sig_a = hasher.signature(a)
        sig_b = hasher.signature(b)
        estimate = estimate_jaccard(sig_a, sig_b)

        assert math.isclose(estimate, exact, abs_tol=0.08)

    def test_signature_length_mismatch_raises(self):
        hasher = MinHasher(num_perm=32)
        sig_a = hasher.signature({"a"})
        other_hasher = MinHasher(num_perm=64)
        sig_b = other_hasher.signature({"a"})
        with pytest.raises(ValueError):
            estimate_jaccard(sig_a, sig_b)


# --------------------------------------------------------------------------- #
# lsh
# --------------------------------------------------------------------------- #

class TestLSH:
    def test_rejects_non_dividing_bands(self):
        with pytest.raises(ValueError):
            LSHIndex(num_perm=100, num_bands=30)

    def test_identical_signatures_are_candidates(self):
        hasher = MinHasher(num_perm=64, seed=5)
        sig = hasher.signature({"a b c", "b c d", "c d e"})

        index = LSHIndex(num_perm=64, num_bands=16)
        index.insert("doc1", sig)
        index.insert("doc2", sig)

        assert "doc2" in index.query_candidates("doc1")
        assert ("doc1", "doc2") in index.candidate_pairs()

    def test_very_different_signatures_are_unlikely_candidates(self):
        hasher = MinHasher(num_perm=128, seed=5)
        sig_a = hasher.signature({f"aaa_{i}" for i in range(50)})
        sig_b = hasher.signature({f"zzz_{i}" for i in range(50)})

        index = LSHIndex(num_perm=128, num_bands=32)
        index.insert("doc1", sig_a)
        index.insert("doc2", sig_b)

        assert "doc2" not in index.query_candidates("doc1")

    def test_s_curve_threshold_matches_known_formula(self):
        # (1/32) ** (1/4) for num_bands=32, rows_per_band=4
        assert math.isclose(s_curve_threshold(32, 4), (1 / 32) ** (1 / 4))

    def test_candidate_probability_bounds(self):
        assert candidate_probability(0.0, 20, 5) == pytest.approx(0.0, abs=1e-9)
        assert candidate_probability(1.0, 20, 5) == pytest.approx(1.0, abs=1e-9)

    def test_len_and_num_buckets(self):
        hasher = MinHasher(num_perm=32, seed=1)
        index = LSHIndex(num_perm=32, num_bands=8)
        index.insert("doc1", hasher.signature({"a b"}))
        index.insert("doc2", hasher.signature({"c d"}))
        assert len(index) == 2
        assert index.num_buckets > 0


# --------------------------------------------------------------------------- #
# simhash
# --------------------------------------------------------------------------- #

class TestSimHash:
    def test_stable_hash_64_deterministic_and_bounded(self):
        h = stable_hash_64("token")
        assert h == stable_hash_64("token")
        assert 0 <= h < 2 ** 64

    def test_fingerprint_bounded(self):
        hasher = TfidfSimHasher(hash_bits=64)
        fp = hasher.fingerprint(["the", "quick", "brown", "fox"])
        assert 0 <= fp < 2 ** 64

    def test_empty_document_fingerprint_is_zero(self):
        hasher = TfidfSimHasher(hash_bits=64)
        assert hasher.fingerprint([]) == 0

    def test_identical_documents_zero_hamming_distance(self):
        tokens = ["renewable", "energy", "solar", "wind", "power"]
        hasher = TfidfSimHasher(hash_bits=64)
        hasher.fit([tokens, tokens])
        fp1 = hasher.fingerprint(tokens)
        fp2 = hasher.fingerprint(tokens)
        assert hamming_distance(fp1, fp2) == 0
        assert hamming_similarity(fp1, fp2) == 1.0

    def test_similar_documents_small_hamming_distance(self):
        tokens_a = "renewable energy sources like solar and wind power".split()
        tokens_b = "renewable energy sources such as solar and wind power".split()
        hasher = TfidfSimHasher(hash_bits=64)
        hasher.fit([tokens_a, tokens_b])
        fp_a = hasher.fingerprint(tokens_a)
        fp_b = hasher.fingerprint(tokens_b)
        # Near-identical bags of words should land close in Hamming space.
        assert hamming_distance(fp_a, fp_b) < 20

    def test_dissimilar_documents_larger_hamming_distance(self):
        tokens_a = "renewable energy sources like solar and wind power".split()
        tokens_b = "cooking pasta requires salted boiling water and timing".split()
        hasher = TfidfSimHasher(hash_bits=64)
        hasher.fit([tokens_a, tokens_b])
        fp_a = hasher.fingerprint(tokens_a)
        fp_b = hasher.fingerprint(tokens_b)
        d_similar_case = 5  # sanity baseline, see test above
        assert hamming_distance(fp_a, fp_b) >= d_similar_case


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

class TestEvaluation:
    def test_confusion_counts(self):
        y_true = [1, 1, 0, 0, 1]
        y_pred = [1, 0, 0, 1, 1]
        counts = confusion_counts(y_true, y_pred)
        assert counts == {"tp": 2, "fp": 1, "fn": 1, "tn": 1}

    def test_precision_recall_f1_known_values(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 0, 0, 0]
        metrics = precision_recall_f1(y_true, y_pred)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 0.5
        assert math.isclose(metrics["f1"], 2 / 3, abs_tol=1e-9)

    def test_precision_recall_f1_handles_degenerate_case(self):
        # No positive predictions at all -> precision defined as 0, not
        # a ZeroDivisionError.
        metrics = precision_recall_f1([1, 1, 0], [0, 0, 0])
        assert metrics["precision"] == 0.0
        assert metrics["recall"] == 0.0
        assert metrics["f1"] == 0.0

    def _toy_pairs_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "text_a": [
                    "renewable energy sources like solar and wind power",
                    "cooking pasta requires salted boiling water",
                    "renewable energy sources such as solar and wind",
                ],
                "text_b": [
                    "renewable energy sources such as solar and wind",
                    "human exploration of mars is an engineering challenge",
                    "cooking pasta requires salted boiling water",
                ],
                "label": [1, 0, 0],
            }
        )

    def test_run_minhash_lsh_pipeline_shapes(self):
        df = self._toy_pairs_df()
        result = run_minhash_lsh_pipeline(df, "text_a", "text_b")
        assert len(result.predictions) == len(df)
        assert all(p in (0, 1) for p in result.predictions)
        assert result.elapsed_seconds >= 0

    def test_run_simhash_pipeline_shapes(self):
        df = self._toy_pairs_df()
        result = run_simhash_pipeline(df, "text_a", "text_b")
        assert len(result.predictions) == len(df)
        assert all(p in (0, 1) for p in result.predictions)

    def test_evaluate_pairs_returns_two_method_rows(self):
        df = self._toy_pairs_df()
        rows, sweep_curves = evaluate_pairs(df, "text_a", "text_b", "label")
        methods = {row["method"] for row in rows}
        assert methods == {"minhash_lsh", "simhash"}
        assert sweep_curves == {}  # auto_threshold defaults to False
        for row in rows:
            assert 0.0 <= row["precision"] <= 1.0
            assert 0.0 <= row["recall"] <= 1.0
            assert 0.0 <= row["f1"] <= 1.0
            assert "threshold_used" in row

    def test_evaluate_pairs_auto_threshold_sweep(self):
        df = self._toy_pairs_df()
        rows, sweep_curves = evaluate_pairs(df, "text_a", "text_b", "label", auto_threshold=True)
        assert set(sweep_curves.keys()) == {"minhash_lsh", "simhash"}
        for row in rows:
            # The chosen threshold must actually appear in that method's sweep curve.
            curve_thresholds = {p["threshold"] for p in sweep_curves[row["method"]]}
            assert row["threshold_used"] in curve_thresholds

    def test_sweep_thresholds_and_best_threshold_row(self):
        y_true = [1, 1, 0, 0]
        scores = [0.9, 0.6, 0.4, 0.1]
        sweep_rows = sweep_thresholds(y_true, scores, thresholds=[0.0, 0.5, 0.7, 1.0])
        assert [r["threshold"] for r in sweep_rows] == [0.0, 0.5, 0.7, 1.0]
        # At threshold 0.5: preds = [1,1,0,0] -> perfect classification.
        perfect = next(r for r in sweep_rows if r["threshold"] == 0.5)
        assert perfect["precision"] == 1.0
        assert perfect["recall"] == 1.0
        best = best_threshold_row(sweep_rows)
        assert best["threshold"] == 0.5
        assert best["f1"] == 1.0


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #

class TestDataset:
    def test_load_corpus_reads_txt_files(self, tmp_path):
        (tmp_path / "doc_a.txt").write_text("Hello world", encoding="utf-8")
        (tmp_path / "doc_b.txt").write_text("Another document", encoding="utf-8")
        (tmp_path / "ignore_me.md").write_text("not a txt file", encoding="utf-8")

        documents = load_corpus(tmp_path)
        assert set(documents.keys()) == {"doc_a.txt", "doc_b.txt"}
        assert documents["doc_a.txt"] == "Hello world"

    def test_load_corpus_missing_folder_raises(self, tmp_path):
        with pytest.raises(NotADirectoryError):
            load_corpus(tmp_path / "does_not_exist")

    def test_load_pairs_csv_roundtrip(self, tmp_path):
        csv_path = tmp_path / "pairs.csv"
        df = pd.DataFrame(
            {
                "q1": ["hello there", "foo bar"],
                "q2": ["hello there!", "baz qux"],
                "label": [1, 0],
            }
        )
        df.to_csv(csv_path, index=False)

        loaded = load_pairs_csv(csv_path, "q1", "q2", label_col="label")
        assert len(loaded) == 2
        assert list(loaded["label"]) == [1, 0]

    def test_load_pairs_csv_respects_limit(self, tmp_path):
        csv_path = tmp_path / "pairs.csv"
        df = pd.DataFrame({"q1": ["a"] * 10, "q2": ["b"] * 10, "label": [0] * 10})
        df.to_csv(csv_path, index=False)

        loaded = load_pairs_csv(csv_path, "q1", "q2", label_col="label", limit=3)
        assert len(loaded) == 3

    def test_save_rows_csv_writes_header_even_when_empty(self, tmp_path):
        out_path = tmp_path / "out" / "empty.csv"
        save_rows_csv([], out_path, fieldnames=["a", "b"])
        assert out_path.exists()
        content = out_path.read_text(encoding="utf-8")
        assert content.strip() == "a,b"
