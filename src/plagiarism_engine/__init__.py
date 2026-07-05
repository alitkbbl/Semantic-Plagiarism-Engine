"""
plagiarism_engine
==================

Semantic Duplicate & Near-Plagiarism Detection Engine.

An educational, industry-inspired implementation of two independent
approaches to near-duplicate document detection:

    1. Shingling + MinHash + LSH   (see ``preprocessing``, ``minhash``, ``lsh``)
    2. TF-IDF weighted SimHash      (see ``simhash``)

Both approaches, and the supporting evaluation harness, are implemented
from scratch on top of the Python standard library, NumPy and Pandas --
no third-party MinHash/SimHash/LSH libraries are used.

See the CLI (``python -m plagiarism_engine.cli --help``) and the
technical report in ``docs/project_spec.pdf`` for full usage and design
details.
"""

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "preprocessing",
    "minhash",
    "lsh",
    "simhash",
    "dataset",
    "evaluation",
]
