"""
simhash.py
==========

From-scratch, TF-IDF weighted SimHash implementation (Charikar, 2002).

Unlike MinHash/LSH, SimHash produces a single fixed-width fingerprint per
document such that similar documents have fingerprints at small Hamming
distance, so a duplicate/near-duplicate test reduces to a bit-count
comparison rather than set operations.

Algorithm
---------
For a document represented as a bag of weighted tokens ``{term: weight}``:

    1. Initialise a 64-dimensional accumulator vector V = 0.
    2. For each term, compute a stable 64-bit hash of the term.
    3. For each of the 64 bit positions:
           if the bit is 1: V[bit] += weight
           else:            V[bit] -= weight
    4. The final fingerprint bit ``i`` is 1 if V[i] > 0, else 0.

The "weight" of a term is its TF-IDF score with respect to the corpus the
document belongs to, computed from scratch below (smoothed IDF, following
the common ``ln((1 + N) / (1 + df)) + 1`` convention so that terms
appearing in every document still get a small positive weight instead of
zero).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Sequence

HASH_BITS = 64


def stable_hash_64(token: str) -> int:
    """Deterministic 64-bit hash of a token, stable across processes.

    Uses BLAKE2b (stdlib, no extra dependency) rather than Python's
    built-in ``hash()``, which is randomized per-process.
    """
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


class TfidfVectorizerFromScratch:
    """Minimal from-scratch TF-IDF computation (no scikit-learn).

    Fit once on a corpus (a list of token lists) to learn document
    frequencies, then compute per-document TF-IDF weight dictionaries.
    """

    def __init__(self):
        self._doc_freq: Dict[str, int] = {}
        self._n_docs: int = 0
        self._fitted = False

    def fit(self, tokenized_docs: Iterable[Sequence[str]]) -> "TfidfVectorizerFromScratch":
        doc_freq: Dict[str, int] = defaultdict(int)
        n_docs = 0
        for tokens in tokenized_docs:
            n_docs += 1
            for term in set(tokens):
                doc_freq[term] += 1
        self._doc_freq = dict(doc_freq)
        self._n_docs = n_docs
        self._fitted = True
        return self

    def _idf(self, term: str) -> float:
        # Smoothed IDF: ln((1 + N) / (1 + df)) + 1.
        # Guarantees a positive weight even for terms in every document,
        # and a sensible fallback (the maximum possible IDF for this
        # corpus) for out-of-vocabulary terms seen only at inference time.
        n = max(self._n_docs, 1)
        df = self._doc_freq.get(term)
        if df is None:
            df = 0  # unseen term: treat as rarer than anything seen
        return math.log((1 + n) / (1 + df)) + 1.0

    def transform(self, tokens: Sequence[str]) -> Dict[str, float]:
        """Return a {term: tf_idf_weight} dict for one document's tokens."""
        if not tokens:
            return {}
        tf_counts = Counter(tokens)
        total_terms = sum(tf_counts.values())
        weights = {}
        for term, count in tf_counts.items():
            tf = count / total_terms
            weights[term] = tf * self._idf(term)
        return weights


class TfidfSimHasher:
    """Computes TF-IDF weighted SimHash fingerprints.

    Parameters
    ----------
    hash_bits:
        Fingerprint width. Fixed at 64 per the project specification.
    ngram_size:
        Size of the token n-grams fed into the hasher. ``1`` (unigrams)
        is the default; larger values make the fingerprint more sensitive
        to local word order, similar to the shingle size used for MinHash.
    """

    def __init__(self, hash_bits: int = HASH_BITS, ngram_size: int = 1):
        self.hash_bits = hash_bits
        self.ngram_size = ngram_size
        self.vectorizer = TfidfVectorizerFromScratch()
        self._fitted = False

    @staticmethod
    def _ngrams(tokens: Sequence[str], n: int) -> List[str]:
        if n <= 1:
            return list(tokens)
        if len(tokens) < n:
            return [" ".join(tokens)] if tokens else []
        return [" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    def fit(self, tokenized_docs: Iterable[Sequence[str]]) -> "TfidfSimHasher":
        """Fit IDF statistics over a corpus of already-tokenized documents."""
        ngram_docs = [self._ngrams(list(tokens), self.ngram_size) for tokens in tokenized_docs]
        self.vectorizer.fit(ngram_docs)
        self._fitted = True
        return self

    def fingerprint(self, tokens: Sequence[str]) -> int:
        """Compute the SimHash fingerprint for one document's tokens.

        If :meth:`fit` was never called, falls back to a single-document
        "corpus" of just this document (equivalent to plain TF weighting),
        which keeps the two-document ``compare`` CLI command usable without
        requiring a full corpus.
        """
        if not self._fitted:
            self.vectorizer.fit([self._ngrams(list(tokens), self.ngram_size)])
            self._fitted = True

        terms = self._ngrams(list(tokens), self.ngram_size)
        weights = self.vectorizer.transform(terms)

        if not weights:
            return 0

        accumulator = [0.0] * self.hash_bits
        for term, weight in weights.items():
            h = stable_hash_64(term)
            for bit in range(self.hash_bits):
                if (h >> bit) & 1:
                    accumulator[bit] += weight
                else:
                    accumulator[bit] -= weight

        fingerprint = 0
        for bit in range(self.hash_bits):
            if accumulator[bit] > 0:
                fingerprint |= 1 << bit
        return fingerprint


def hamming_distance(a: int, b: int) -> int:
    """Number of differing bits between two integer fingerprints."""
    return bin(a ^ b).count("1")


def hamming_similarity(a: int, b: int, bits: int = HASH_BITS) -> float:
    """Normalized similarity in [0, 1] derived from Hamming distance."""
    if bits == 0:
        return 1.0
    return 1.0 - (hamming_distance(a, b) / bits)
