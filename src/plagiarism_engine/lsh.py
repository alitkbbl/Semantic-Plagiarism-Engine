"""
lsh.py
======

From-scratch Locality-Sensitive Hashing (LSH) over MinHash signatures,
using the classic "banding" technique (Leskovec, Rajaraman & Ullman,
*Mining of Massive Datasets*, ch. 3).

Idea
----
A MinHash signature of length ``num_perm`` is split into ``num_bands``
contiguous "bands" of ``rows_per_band = num_perm / num_bands`` rows each.
Each band is hashed to a bucket. Two documents are considered *candidate*
duplicates if **any** of their bands hash to the same bucket.

Why this reduces work
----------------------
Instead of comparing every one of the O(n^2) document pairs, we only ever
compare documents that collide in at least one band bucket. In the worst
case every document could still land in one giant bucket together, but in
practice -- and by construction, via the S-curve argument below -- only
genuinely similar documents are likely to collide, so the number of
candidate pairs is typically a small fraction of n^2 (quantified for the
sample corpus in the technical report).

The banding scheme approximates a step function ("S-curve") in the
probability of becoming a candidate pair as a function of true Jaccard
similarity ``s``:

    P(candidate) = 1 - (1 - s**rows_per_band) ** num_bands

which is approximately 0 for s below a threshold and approximately 1
above it, with the threshold near ``(1 / num_bands) ** (1 / rows_per_band)``.
This lets us trade off false negatives vs. false positives by choosing
``num_bands`` and ``rows_per_band`` (see :func:`s_curve_threshold`).
"""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from typing import Dict, Hashable, Iterable, List, Set, Tuple

import numpy as np


def _hash_band(band_values: Iterable[int], band_index: int) -> str:
    """Hash one band's worth of signature rows to a bucket id.

    The band index is mixed into the hash so that identical row values
    occurring in *different* bands never collide with each other (each
    band gets its own independent bucket space).
    """
    hasher = hashlib.sha1()
    hasher.update(band_index.to_bytes(4, byteorder="big"))
    for v in band_values:
        hasher.update(int(v).to_bytes(8, byteorder="big"))
    return hasher.hexdigest()


def s_curve_threshold(num_bands: int, rows_per_band: int) -> float:
    """Approximate Jaccard similarity threshold at which the LSH S-curve
    is at its steepest, commonly used as a rule-of-thumb "detection
    threshold": ``(1 / num_bands) ** (1 / rows_per_band)``.
    """
    return (1.0 / num_bands) ** (1.0 / rows_per_band)


def candidate_probability(s: float, num_bands: int, rows_per_band: int) -> float:
    """Exact probability that two documents with true Jaccard similarity
    ``s`` become LSH candidates, per the banding S-curve formula."""
    return 1.0 - (1.0 - s ** rows_per_band) ** num_bands


class LSHIndex:
    """An in-memory banded LSH index over MinHash signatures.

    Parameters
    ----------
    num_perm:
        Length of the MinHash signatures that will be inserted.
    num_bands:
        Number of bands to split each signature into. Must evenly divide
        ``num_perm``.
    """

    def __init__(self, num_perm: int = 128, num_bands: int = 32):
        if num_perm % num_bands != 0:
            raise ValueError(
                f"num_perm ({num_perm}) must be evenly divisible by "
                f"num_bands ({num_bands})"
            )
        self.num_perm = num_perm
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands

        # (band_index, bucket_hash) -> set of document ids in that bucket.
        self._buckets: Dict[Tuple[int, str], Set[Hashable]] = defaultdict(set)
        self._signatures: Dict[Hashable, np.ndarray] = {}

    def insert(self, doc_id: Hashable, signature: np.ndarray) -> None:
        """Insert a document's MinHash signature into the index."""
        if len(signature) != self.num_perm:
            raise ValueError(
                f"Signature length {len(signature)} does not match "
                f"index num_perm {self.num_perm}"
            )
        self._signatures[doc_id] = signature
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_values = signature[start:end]
            bucket_key = (band_idx, _hash_band(band_values, band_idx))
            self._buckets[bucket_key].add(doc_id)

    def query_candidates(self, doc_id: Hashable) -> Set[Hashable]:
        """Return the set of candidate-duplicate document ids for a
        document already present in the index (excluding itself)."""
        signature = self._signatures[doc_id]
        candidates: Set[Hashable] = set()
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band_values = signature[start:end]
            bucket_key = (band_idx, _hash_band(band_values, band_idx))
            candidates.update(self._buckets[bucket_key])
        candidates.discard(doc_id)
        return candidates

    def candidate_pairs(self) -> Set[Tuple[Hashable, Hashable]]:
        """Return all candidate pairs across the whole index.

        Each pair is returned once, as a tuple sorted for determinism.
        This is the set of pairs that a downstream exact-Jaccard pass
        actually needs to verify -- typically far smaller than the full
        O(n^2) set of all possible pairs.
        """
        pairs: Set[Tuple[Hashable, Hashable]] = set()
        for members in self._buckets.values():
            if len(members) < 2:
                continue
            for a, b in itertools.combinations(sorted(members, key=str), 2):
                pairs.add((a, b))
        return pairs

    def __len__(self) -> int:
        return len(self._signatures)

    @property
    def num_buckets(self) -> int:
        return len(self._buckets)
