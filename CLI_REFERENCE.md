# Semantic Plagiarism Engine — Complete CLI Reference

A structured reference of **every command and flag** available in the project's CLI
(`plagiarism_engine.cli`) and its companion data-prep script
(`scripts/prepare_pan_pc11_pairs.py`). This file is a standalone reference — it is not part of the
project itself.

---

## 0. Setup (run once)

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```
*(Windows: `.venv\Scripts\activate`)*

```bash
pip install -r requirements.txt -e .
```

```bash
pytest -v
```
Expected: `46 passed`.

> If you've downloaded/cloned this project more than once on the same machine, make sure you `pip
> install -e .` from the copy you're actually running commands in — an editable install links to a
> specific folder, and Python will silently import from the wrong copy otherwise:
> ```bash
> pip uninstall -y plagiarism-engine
> pip install --no-build-isolation -e .
> ```

---

## 1. `compare` — compare two documents directly

**Purpose:** run both detection backends (Shingling+MinHash+LSH and TF-IDF SimHash) on exactly two
documents and print/save the comparison.

### Full syntax
```bash
python -m plagiarism_engine.cli compare \
    --file-a <path> \
    --file-b <path> \
    [--shingle-size 3] \
    [--num-perm 128] \
    [--num-bands 32] \
    [--hash-bits 64] \
    [--output <path.json>]
```

### Flags
| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--file-a` | ✅ | — | Path to the first document |
| `--file-b` | ✅ | — | Path to the second document |
| `--shingle-size` | | `3` | Word shingle size ($k$) for MinHash/LSH |
| `--num-perm` | | `128` | MinHash signature length |
| `--num-bands` | | `32` | Number of LSH bands (must evenly divide `--num-perm`) |
| `--hash-bits` | | `64` | SimHash fingerprint width in bits |
| `--output` | | — | Optional path to also write a JSON result |

### Examples
```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt \
    --file-b data/sample_corpus/doc_02.txt
```

```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt \
    --file-b data/sample_corpus/doc_03.txt \
    --output outputs/two_file_compare.json
```

### What it prints
Exact Jaccard similarity, MinHash-estimated similarity, whether the pair is an LSH candidate, SimHash
Hamming distance, and SimHash Hamming similarity.

---

## 2. `corpus` — search a folder for near-duplicate documents

**Purpose:** scan every `*.txt` file in a folder, build a MinHash+LSH index, and report near-duplicate
pairs without doing an exhaustive $O(n^2)$ comparison.

### Full syntax
```bash
python -m plagiarism_engine.cli corpus \
    --data <folder> \
    [--threshold 0.25] \
    [--shingle-size 3] \
    [--num-perm 128] \
    [--num-bands 32] \
    --output <path.csv>
```

### Flags
| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--data` | ✅ | — | Folder containing `*.txt` documents |
| `--threshold` | | `0.25` | Minimum exact-Jaccard similarity to report a pair |
| `--shingle-size` | | `3` | Word shingle size ($k$) |
| `--num-perm` | | `128` | MinHash signature length |
| `--num-bands` | | `32` | Number of LSH bands (must evenly divide `--num-perm`) |
| `--output` | ✅ | — | Path to write the candidates CSV |

### Example
```bash
python -m plagiarism_engine.cli corpus \
    --data data/sample_corpus \
    --threshold 0.25 \
    --shingle-size 3 \
    --output outputs/candidates.csv
```

### What it prints
Number of documents scanned, total possible pairs ($O(n^2)$), number of LSH candidate pairs, percentage
of comparisons avoided by LSH, number of pairs above `--threshold`, and timing breakdown (indexing / LSH
lookup / exact verification).

### Output CSV columns
`doc_a, doc_b, exact_jaccard_similarity, minhash_estimated_similarity`

---

## 3. `pairs` — evaluate on a labeled pair dataset

**Purpose:** run **both** backends over a CSV of labeled document pairs and report
precision/recall/F1/timing for each.

### Full syntax
```bash
python -m plagiarism_engine.cli pairs \
    --pairs <path.csv> \
    --text-col-a <column> \
    --text-col-b <column> \
    --label-col <column> \
    [--limit N] \
    [--shingle-size 3] \
    [--num-perm 128] \
    [--num-bands 32] \
    [--minhash-threshold 0.5] \
    [--hash-bits 64] \
    [--simhash-threshold 0.85] \
    [--sweep] \
    [--sweep-output <path.csv>] \
    --output <path.csv>
```

### Flags
| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--pairs` | ✅ | — | CSV file of labeled document pairs |
| `--text-col-a` | ✅ | — | Column name of the first document's text |
| `--text-col-b` | ✅ | — | Column name of the second document's text |
| `--label-col` | ✅ | — | Column name of the binary label (1 = duplicate, 0 = not) |
| `--limit` | | — | Only evaluate the first N rows |
| `--shingle-size` | | `3` | Word shingle size ($k$) for the MinHash+LSH pipeline |
| `--num-perm` | | `128` | MinHash signature length |
| `--num-bands` | | `32` | Number of LSH bands (must evenly divide `--num-perm`) |
| `--minhash-threshold` | | `0.5` | MinHash similarity cutoff for "duplicate" — **ignored if `--sweep` is set** |
| `--hash-bits` | | `64` | SimHash fingerprint width in bits |
| `--simhash-threshold` | | `0.85` | SimHash Hamming-similarity cutoff — **ignored if `--sweep` is set** |
| `--sweep` | | off | Auto-select each method's F1-maximizing threshold instead of using the fixed ones above |
| `--sweep-output` | | — | Also save the full threshold/precision/recall/F1 curve (only meaningful with `--sweep`) |
| `--output` | ✅ | — | Path to write the metrics CSV |

### Examples

**Fixed thresholds (manual):**
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/quora/sample_pairs.csv \
    --text-col-a question1 --text-col-b question2 --label-col is_duplicate \
    --shingle-size 1 --minhash-threshold 0.13 --simhash-threshold 0.46 \
    --output outputs/metrics.csv
```

**Automatic threshold selection (recommended — no guessing):**
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/quora/sample_pairs.csv \
    --text-col-a question1 --text-col-b question2 --label-col is_duplicate \
    --shingle-size 1 \
    --sweep --sweep-output outputs/threshold_sweep.csv \
    --output outputs/metrics.csv
```

**Limiting rows (useful for a quick check on a huge dataset):**
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/quora/train.csv \
    --text-col-a question1 --text-col-b question2 --label-col is_duplicate \
    --limit 5000 --sweep \
    --output outputs/metrics.csv
```

### When to use `--sweep`
Use it whenever you evaluate on a **new** dataset. A fixed threshold tuned for one kind of text (e.g.
short questions) can silently fail on another (e.g. long literary spans) — the symptom is **very high
precision with collapsed recall**, which looks like a broken algorithm but is actually just a
miscalibrated cutoff. `--sweep` costs almost nothing extra: similarity scores are computed once and
re-thresholded across a grid (0.00–1.00, step 0.02) cheaply.

### Output CSV columns
`method, num_pairs, threshold_used, precision, recall, f1, true_positive, false_positive, false_negative, true_negative, total_time_seconds, avg_time_per_pair_ms, lsh_candidate_rate` *(the last column is MinHash+LSH only)*

### `--sweep-output` CSV columns
`method, threshold, precision, recall, f1, tp, fp, fn, tn` (one row per threshold tested, per method)

---

## 4. PAN-PC-11 data-prep script (`scripts/prepare_pan_pc11_pairs.py`)

**Purpose:** convert a locally-downloaded copy of the real **PAN-PC-11** plagiarism corpus
([Zenodo, DOI 10.5281/zenodo.3250095](https://zenodo.org/records/3250095)) — a folder of raw `.txt`
documents plus `.xml` ground-truth annotations — into a `pairs`-compatible CSV. This script is
standalone; it does not require or modify anything in `src/plagiarism_engine/`.

> **Prerequisite:** download and extract PAN-PC-11 yourself first. Point every command below at
> `.../pan-plagiarism-corpus-2011/external-detection-corpus` specifically — not the corpus root, and not
> `intrinsic-detection-corpus` (which has no source documents to pair against at all).

### Full syntax
```bash
python scripts/prepare_pan_pc11_pairs.py \
    --corpus-dir <path> \
    [--output <path.csv>] \
    [--limit-positive 2000] \
    [--num-negatives N] \
    [--min-chars 20] \
    [--seed 42] \
    [--feature-name plagiarism] \
    [--inspect] \
    [--max-files 3000]
```

### Flags
| Flag | Required | Default | Meaning |
|---|---|---|---|
| `--corpus-dir` | ✅ | — | Root of the extracted corpus (point at `external-detection-corpus`) |
| `--output` | ✅ unless `--inspect` | — | Where to write the pairs CSV |
| `--limit-positive` | | `2000` | Cap on the number of positive (plagiarism) pairs extracted |
| `--num-negatives` | | = positive count | Number of negative pairs to build |
| `--min-chars` | | `20` | Discard spans shorter than this many characters |
| `--seed` | | `42` | Random seed (for reproducible negative sampling) |
| `--feature-name` | | `plagiarism` | The `<feature name="...">` value marking a plagiarism case |
| `--inspect` | | off | Scan and report every real `<feature name="...">` value found, then exit |
| `--max-files` | | `3000` | Cap on how many annotation files `--inspect` scans (speed on huge corpora) |

### Step-by-step workflow

**Step 1 — confirm the annotation schema on YOUR download** (always do this first):
```bash
python scripts/prepare_pan_pc11_pairs.py \
    --corpus-dir data/raw/pan-plagiarism-corpus-2011/external-detection-corpus \
    --inspect
```
Look for a `<feature name="...">` value with `this_offset` / `this_length` / `source_reference` /
`source_offset` / `source_length` attributes — that's the one to use in Step 2. On the confirmed 2011
release this value is `plagiarism` (also flagged automatically by the tool). Values like `about` or
`md5Hash` are source-document metadata, not plagiarism cases — ignore those.

**Step 2 — build the labeled pairs CSV:**
```bash
python scripts/prepare_pan_pc11_pairs.py \
    --corpus-dir data/raw/pan-plagiarism-corpus-2011/external-detection-corpus \
    --output data/processed/pan_pc11_pairs.csv \
    --feature-name plagiarism
```
(Add `--limit-positive` / `--num-negatives` / `--min-chars` to change the sample size.)

**Step 3 — evaluate with the ordinary `pairs` command** (use `--shingle-size 1`; PAN-PC-11's obfuscated
spans destroy larger contiguous shingles — see below):
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/processed/pan_pc11_pairs.csv \
    --text-col-a text_a --text-col-b text_b --label-col label \
    --shingle-size 1 \
    --sweep --sweep-output outputs/pan_pc11_threshold_sweep.csv \
    --output outputs/metrics_pan_pc11.csv
```

### Why `--shingle-size 1` for PAN-PC-11 specifically
| Config | MinHash+LSH F1 |
|---|---|
| `--shingle-size 3` (default), fixed thresholds | 0.070 |
| `--shingle-size 3`, `--sweep` | 0.667 (trivial "predict all" baseline — no real signal at k=3) |
| `--shingle-size 1`, `--sweep` | **0.920** |

PAN-PC-11's obfuscated plagiarism cases scatter word substitutions throughout long passages; a single
substitution destroys every overlapping 3-word shingle, while unigram (word-level) overlap survives far
better.

### What `--inspect` prints
Total `(txt, xml)` file pairs found, a breakdown of suspicious-document vs. source-document counts, every
distinct `<feature name="...">` value found with its occurrence count and an example file, and (for the
value that looks like a plagiarism case) a few example parsed entries.

### What building the CSV prints
Number of positive pairs built, how many were skipped (missing source document / too short) and how many
XML files failed to parse, number of negative pairs built (and how many document pairs were excluded from
negative sampling because they have a real plagiarism relationship elsewhere in the corpus), and the total
row count written.

### Output CSV columns
`text_a, text_b, label, doc_a, doc_b`

---

## 5. Quick lookup — every command in one place

```bash
# Install
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt -e .

# Test
pytest -v

# 1) Compare two documents
python -m plagiarism_engine.cli compare --file-a <fileA> --file-b <fileB> [--output out.json]

# 2) Search a folder for near-duplicates
python -m plagiarism_engine.cli corpus --data <folder> --threshold 0.25 --output candidates.csv

# 3) Evaluate on a labeled pair CSV
python -m plagiarism_engine.cli pairs \
    --pairs <pairs.csv> --text-col-a <col> --text-col-b <col> --label-col <col> \
    [--shingle-size 1] [--sweep] [--sweep-output sweep.csv] --output metrics.csv

# PAN-PC-11 only: inspect schema
python scripts/prepare_pan_pc11_pairs.py --corpus-dir <external-detection-corpus> --inspect

# PAN-PC-11 only: build pairs CSV
python scripts/prepare_pan_pc11_pairs.py --corpus-dir <external-detection-corpus> \
    --output data/processed/pan_pc11_pairs.csv --feature-name plagiarism

# PAN-PC-11 only: evaluate
python -m plagiarism_engine.cli pairs \
    --pairs data/processed/pan_pc11_pairs.csv --text-col-a text_a --text-col-b text_b --label-col label \
    --shingle-size 1 --sweep --output outputs/metrics_pan_pc11.csv

# Any command's full flag list
python -m plagiarism_engine.cli <compare|corpus|pairs> --help
python scripts/prepare_pan_pc11_pairs.py --help
```
