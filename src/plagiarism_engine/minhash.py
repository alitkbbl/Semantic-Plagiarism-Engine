"""
minhash.py
==========

From-scratch MinHash implementation.

No third-party MinHash/LSH libraries (e.g. ``datasketch``) are used here --
only the Python standard library (``hashlib`` for a stable base hash) and
NumPy for compact signature storage.

Background
----------
Given two sets A and B, the Jaccard similarity is::

    J(A, B) = |A intersect B| / |A union B|

Computing J exactly for every pair in a large collection is O(n^2) in the
number of documents (see the technical report, section "Complexity of
exact pairwise comparison"). MinHash lets us *estimate* J(A, B) from two
short signature vectors instead of the full shingle sets, and -- combined
with LSH (see ``lsh.py``) -- lets us avoid ever comparing most pairs at all.

Theory
------
For a family of independent, uniformly-random hash functions h, it is a
classical result that::

    P[ min_{x in A} h(x) == min_{x in B} h(x) ] == J(A, B)

So if we draw ``num_perm`` independent hash functions and, for each one,
record the minimum hash value over a set's elements, the *fraction* of
positions at which two sets' signatures agree is an unbiased estimator of
their true Jaccard similarity, with estimation error that shrinks as
O(1 / sqrt(num_perm)).

We approximate "independent random hash functions over a fixed universe"
using the standard universal-hashing family::

    h_i(x) = (a_i * x + b_i) mod p

for a large prime ``p`` and random coefficients ``a_i, b_i`` drawn once at
``MinHasher`` construction time, where ``x`` is a stable 64-bit hash of a
shingle string produced via SHA-1 (Python's built-in ``hash()`` is
deliberately randomized per-process for security reasons, so it cannot be
used here -- we need the same shingle to hash identically across runs and
across documents).
"""

from __future__ import annotations

import hashlib
import random
from typing import Iterable, List

import numpy as np

# 2^61 - 1, a Mersenne prime comfortably larger than any 64-bit shingle hash,
# used as the modulus for the universal hash family.
_MERSENNE_PRIME = (1 << 61) - 1
_MAX_HASH = (1 << 64) - 1


def stable_hash(token: str) -> int:
    """Deterministic 64-bit hash of a string, stable across processes.

    Implemented with SHA-1 rather than Python's built-in ``hash()`` because
    the latter is randomized per-interpreter-run (PYTHONHASHSEED) and would
    make MinHash signatures incomparable across separate CLI invocations.
    """
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


class MinHasher:
    """Generates MinHash signatures for shingle sets.

    Parameters
    ----------
    num_perm:
        Number of hash functions / signature length. 128 or 256 are the
        values recommended by the project specification: 128 gives a
        typical estimation standard error of about 1/sqrt(128) ~= 8.8%,
        256 roughly halves that to ~6.25%, at double the signature size
        and computation cost.
    seed:
        Random seed used to draw the universal hash family coefficients,
        kept fixed by default so that signatures are reproducible.
    """

    def __init__(self, num_perm: int = 128, seed: int = 42):
        if num_perm < 1:
            raise ValueError("num_perm must be >= 1")
        self.num_perm = num_perm
        self.seed = seed

        rng = random.Random(seed)
        # a must be non-zero mod p for h to be a valid universal hash.
        self._a: List[int] = [rng.randrange(1, _MERSENNE_PRIME) for _ in range(num_perm)]
        self._b: List[int] = [rng.randrange(0, _MERSENNE_PRIME) for _ in range(num_perm)]

    def signature(self, shingles: Iterable[str]) -> np.ndarray:
        """Compute the MinHash signature for a set of shingles.

        Returns a ``numpy.ndarray`` of ``dtype=uint64`` and length
        ``num_perm``. An empty shingle set (empty document) yields a
        sentinel signature of all ``_MAX_HASH`` values; two empty
        documents therefore compare as identical (similarity 1.0), which
        matches the conventional definition J(empty, empty) = 1.
        """
        shingles = list(shingles)
        sig = [_MAX_HASH] * self.num_perm
        if not shingles:
            return np.array(sig, dtype=np.uint64)

        hashed_values = [stable_hash(s) for s in shingles]
        for i in range(self.num_perm):
            a, b = self._a[i], self._b[i]
            min_hash = min((a * hv + b) % _MERSENNE_PRIME for hv in hashed_values)
            sig[i] = min_hash & _MAX_HASH
        return np.array(sig, dtype=np.uint64)


def estimate_jaccard(sig_a: np.ndarray, sig_b: np.ndarray) -> float:
    """Estimate Jaccard similarity from two MinHash signatures.

    This is simply the fraction of signature positions at which the two
    signatures agree, per the MinHash theorem described in the module
    docstring.
    """
    if len(sig_a) != len(sig_b):
        raise ValueError("Signatures must have the same length to compare")
    if len(sig_a) == 0:
        return 0.0
    return float(np.mean(sig_a == sig_b))
