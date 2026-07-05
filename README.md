# Semantic Duplicate & Near-Plagiarism Detection Engine

An educational, industry-inspired command line system for detecting duplicate, copied, and paraphrased
documents. It implements and compares **two independent similarity-detection approaches, both from
scratch**:

1. **Shingling + MinHash + LSH** -- documents as sets of word $k$-shingles; Jaccard similarity estimated
   via MinHash signatures and pruned via Locality-Sensitive Hashing so a large corpus never needs
   exhaustive pairwise comparison.
2. **TF-IDF weighted SimHash** -- each document reduced to a single 64-bit fingerprint; similarity is a
   Hamming-distance bit-count.

No third-party MinHash/SimHash/LSH library (e.g. `datasketch`) is used for the core algorithms -- only the
Python standard library, NumPy, and Pandas. This is a **CLI-only** project: no graphical interface, no
terminal UI.

📄 Full technical report (method, parameter selection, datasets, results, error analysis):
[`docs/project_spec.pdf`](docs/project_spec.pdf)
📓 Interactive walkthrough of every method and CLI command: [`notebooks/exploration.ipynb`](notebooks/exploration.ipynb)

---

## Results at a glance

Validated on three datasets -- a hand-curated document corpus, a synthetic labeled question-pair set, and
the **real, 22,186-pair PAN-PC-11 plagiarism corpus**:

| Evaluation | Method | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| Sample corpus (11 docs) -- LSH candidate generation | -- | -- | -- | **94.55% of pairwise comparisons avoided**, 0 true near-duplicates missed |
| Synthetic question pairs (60 pairs) | MinHash+LSH | 0.730 | 0.900 | 0.806 |
| Synthetic question pairs (60 pairs) | TF-IDF SimHash | 0.556 | 1.000 | 0.714 |
| Real PAN-PC-11 (4,000 pairs, tuned) | MinHash+LSH | 0.971 | 0.875 | **0.920** |
| Real PAN-PC-11 (4,000 pairs, tuned) | TF-IDF SimHash | 0.839 | 0.745 | 0.789 |

Neither method is categorically better -- SimHash wins on short questions, MinHash+LSH wins on long,
heavily-obfuscated literary spans once the shingle size and decision threshold are tuned to the corpus (see
`docs/project_spec.pdf`, Sections 5 and 7, for the full tuning story and an important caveat on the
PAN-PC-11 numbers).

---

## Installation

Requires Python 3.9+.

```bash
python -m venv .venv
```
```bash
# Linux / macOS
source .venv/bin/activate

# Windows (PowerShell / CMD)
.venv\Scripts\activate

```
```bash
pip install -e ".[dev]" 
```

This installs the runtime dependencies (`numpy`, `pandas`, `click`), `pytest` for the test suite, and the
`plagiarism_engine` package itself in editable mode. A console entry point `plagiarism-engine` is also
installed; it is exactly equivalent to `python -m plagiarism_engine.cli`.

> **Already have another copy of this project installed?** `pip install -e .` registers an editable link
> to whichever folder you run it from. If you've downloaded this project more than once, re-run
> `pip uninstall -y plagiarism-engine && pip install --no-build-isolation -e .` from the copy you're
> actually working in, or Python may silently import code from a different, older copy on disk.

---

## Quickstart

```bash
# 1. Compare two documents directly
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt --file-b data/sample_corpus/doc_02.txt

# 2. Search a folder for near-duplicates
python -m plagiarism_engine.cli corpus \
    --data data/sample_corpus --threshold 0.25 --output outputs/candidates.csv

# 3. Evaluate on a labeled pair dataset (bundled synthetic demo file)
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/quora/sample_pairs.csv \
    --text-col-a question1 --text-col-b question2 --label-col is_duplicate \
    --shingle-size 1 --sweep --output outputs/metrics.csv
```

---

## Execution instructions

The CLI exposes three commands: `compare`, `corpus`, and `pairs`.

### 1. `compare` -- compare two documents

```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt \
    --file-b data/sample_corpus/doc_02.txt \
    --output outputs/two_file_compare.json
```

Prints exact Jaccard similarity, the MinHash-estimated similarity, whether the pair would be flagged as an
LSH candidate, and the SimHash Hamming distance/similarity, and (optionally) writes a JSON report.

### 2. `corpus` -- search a folder for near-duplicate documents

```bash
python -m plagiarism_engine.cli corpus \
    --data data/sample_corpus \
    --threshold 0.25 \
    --shingle-size 3 \
    --output outputs/candidates.csv
```

Builds a MinHash+LSH index over every `*.txt` file in `--data`, uses LSH to cut the number of expensive
exact-Jaccard comparisons down from the full O(n²) pair set to just the LSH candidate pairs, verifies each
candidate with exact Jaccard, and writes the pairs at or above `--threshold` to `--output`. Prints a
summary of how many comparisons LSH avoided (94.55% on the bundled sample corpus).

### 3. `pairs` -- evaluate on a labeled pair dataset

```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/quora/sample_pairs.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --limit 5000 \
    --output outputs/metrics.csv
```

Runs **both** backends over every row of a labeled pair CSV (Quora Question Pairs, Stack Exchange
Duplicates, a PAN-PC-11-derived pair list -- see [below](#using-the-real-pan-pc-11-corpus) -- or the small
synthetic demo file shipped at `data/raw/quora/sample_pairs.csv`) and reports precision, recall, F1, and
execution time for each, written to `--output`.

Similarity-score distributions vary a lot by document length and by how heavily a dataset's positive pairs
are paraphrased/obfuscated, so a fixed decision threshold tuned for one dataset can perform very poorly on
another -- high precision with collapsed recall is the signature of this, not necessarily a broken
similarity measure (see `docs/project_spec.pdf`, Section 5.6). Add `--sweep` to have each method's
threshold chosen automatically to maximize F1 on the given dataset instead of guessing (this costs almost
nothing extra -- similarity scores are computed once and swept cheaply); add `--sweep-output <path>` to
also save the full threshold/precision/recall/F1 curve.

Run `python -m plagiarism_engine.cli <command> --help` for the full list of tunable options (shingle size,
number of MinHash permutations, number of LSH bands, SimHash bit width, decision thresholds, etc.) on any
command.

### Using the real PAN-PC-11 corpus

`scripts/prepare_pan_pc11_pairs.py` converts a locally-downloaded copy of the real PAN-PC-11 corpus
(raw documents + XML plagiarism annotations, [Zenodo](https://zenodo.org/records/3250095)) into a
`pairs`-compatible CSV, with no changes needed to `src/plagiarism_engine/`:

```bash
# 1. Confirm the annotation schema against your actual download:
python scripts/prepare_pan_pc11_pairs.py \
    --corpus-dir data/raw/pan-plagiarism-corpus-2011/external-detection-corpus --inspect

# 2. Build the labeled pairs CSV:
python scripts/prepare_pan_pc11_pairs.py \
    --corpus-dir data/raw/pan-plagiarism-corpus-2011/external-detection-corpus \
    --output data/processed/pan_pc11_pairs.csv --feature-name plagiarism

# 3. Evaluate (k=1 recommended -- see docs/project_spec.pdf, Section 5.7, for why):
python -m plagiarism_engine.cli pairs \
    --pairs data/processed/pan_pc11_pairs.csv \
    --text-col-a text_a --text-col-b text_b --label-col label \
    --shingle-size 1 --sweep --output outputs/metrics_pan_pc11.csv
```

See `data/raw/README.md` for the full walkthrough, the confirmed real directory layout, and a documented
limitation of this dataset construction (length mismatch between positive and negative pairs).

---

## Running the tests

```bash
pip install -r requirements.txt -e .   # if not already installed
pytest -v
```

The suite in `tests/test_engine.py` (46 tests) covers preprocessing (including the empty-document,
short-document, and unusual-character edge cases), MinHash signature correctness and determinism, LSH
candidate generation, SimHash fingerprinting, evaluation metrics (including the threshold-sweep utilities),
and dataset I/O.

---

## Project structure

```
semantic-plagiarism-engine/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── docs/
│   ├── project_spec.tex        # Technical report (LaTeX source)
│   ├── project_spec.pdf        # Technical report (compiled)
│   └── figures/                # Report figures (generated, reproducible)
├── data/
│   ├── sample_corpus/          # Small demo corpus for `compare` / `corpus`
│   ├── raw/                    # Large third-party datasets go here (not bundled)
│   │   └── quora/sample_pairs.csv   # Small synthetic demo pairs (bundled)
│   └── processed/              # Derived/cached datasets go here
├── src/
│   └── plagiarism_engine/
│       ├── __init__.py
│       ├── preprocessing.py    # Normalization, tokenization, shingling
│       ├── minhash.py          # From-scratch MinHash
│       ├── lsh.py              # From-scratch banded LSH
│       ├── simhash.py          # From-scratch TF-IDF weighted SimHash
│       ├── dataset.py          # Corpus / pair-CSV loading and saving
│       ├── evaluation.py       # Metrics, threshold sweep, pipeline runners
│       └── cli.py              # `compare` / `corpus` / `pairs` commands
├── scripts/
│   └── prepare_pan_pc11_pairs.py   # Convert a local PAN-PC-11 download into
│                                    # a `pairs`-compatible CSV (optional)
├── notebooks/
│   └── exploration.ipynb       # Full walkthrough: all 3 CLI commands, parameter
│                                # selection, and the real PAN-PC-11 case study
├── tests/
│   └── test_engine.py          # 46 tests
└── outputs/
    ├── metrics.csv             # Written by `pairs`
    └── candidates.csv          # Written by `corpus`
```

---

## Method summary

| | Shingling + MinHash + LSH | TF-IDF weighted SimHash |
|---|---|---|
| Document representation | Set of word $k$-shingles | Weighted bag of tokens |
| Similarity signal | Jaccard similarity (exact or estimated) | Hamming distance between 64-bit fingerprints |
| Sensitive to | Exact wording / word order | Vocabulary overlap (order-independent) |
| Scales via | LSH banding (skip most pairwise comparisons) | O(1) fingerprint comparison per pair |
| Strength | Robust when edits are concentrated (verbatim / lightly-edited copies) | Robust to reordering; strong on very short text |
| Weakness | Struggles when edits are scattered throughout a long passage, unless $k$ is reduced | Still lexical, not truly semantic; sensitive to vocabulary swaps |

See `docs/project_spec.pdf` for the full write-up: why exact pairwise Jaccard comparison is O(n²), how
LSH's banding scheme trades off false positives/negatives, parameter selection for both pipelines
(including the automatic threshold-sweep methodology), and a worked error analysis of specific
misclassified pairs from both the synthetic and real-corpus experiments.

---

## Recommended datasets for larger-scale evaluation

* **PAN-PC-11** (PAN 2011 Plagiarism Corpus) -- primary recommended dataset, available via
  [Zenodo](https://zenodo.org/records/3250095). It's a raw document + XML-annotation corpus rather than a
  simple CSV; see [above](#using-the-real-pan-pc-11-corpus), `data/raw/README.md`, and
  `scripts/prepare_pan_pc11_pairs.py`.
* **Stack Exchange Duplicates** -- alternative, available on Hugging Face.
* **Quora Question Pairs** -- alternative, 400,000+ labeled question pairs, available on Hugging Face /
  Kaggle.

None of the large datasets are bundled with the project (see `data/raw/README.md`); only the small
synthetic demo file is included so `pairs` is runnable without a network connection.
