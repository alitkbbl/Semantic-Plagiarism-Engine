# Semantic Plagiarism Engine — Copy-Paste CLI Reference

Every command below is **complete and ready to run as-is** — no placeholders, no `<...>` to fill in.
They use the real files that already exist in the project. Run them in order from the project root.

---

## 0. Setup

```bash
python -m venv .venv
```

```bash
source .venv/bin/activate
```

```bash
pip install -r requirements.txt -e .
```

```bash
pytest -v
```

Expected last line: `46 passed`.

---

## 1. `compare` — compare two documents directly

### Example 1 — verbatim copy (should show similarity = 1.0)
```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt \
    --file-b data/sample_corpus/doc_03.txt
```

### Example 2 — heavily paraphrased pair (should show low Jaccard, moderate SimHash)
```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt \
    --file-b data/sample_corpus/doc_02.txt
```

### Example 3 — lightly-edited near-duplicate, saved to a JSON file
```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt \
    --file-b data/sample_corpus/doc_11.txt \
    --output outputs/two_file_compare.json
```

```bash
cat outputs/two_file_compare.json
```

### Example 4 — same comparison with custom parameters (bigger MinHash signature, wider SimHash)
```bash
python -m plagiarism_engine.cli compare \
    --file-a data/sample_corpus/doc_01.txt \
    --file-b data/sample_corpus/doc_02.txt \
    --shingle-size 2 \
    --num-perm 256 \
    --num-bands 64 \
    --hash-bits 64
```

### See every available option
```bash
python -m plagiarism_engine.cli compare --help
```

---

## 2. `corpus` — search a folder for near-duplicate documents

### Run it
```bash
python -m plagiarism_engine.cli corpus \
    --data data/sample_corpus \
    --threshold 0.25 \
    --shingle-size 3 \
    --output outputs/candidates.csv
```

### Check the result
```bash
cat outputs/candidates.csv
```
Expected 3 rows, including `doc_01.txt,doc_03.txt,1.0,1.0`.

### Same thing with a stricter threshold (fewer/no matches expected)
```bash
python -m plagiarism_engine.cli corpus \
    --data data/sample_corpus \
    --threshold 0.6 \
    --shingle-size 3 \
    --output outputs/candidates_strict.csv
```

```bash
cat outputs/candidates_strict.csv
```

### See every available option
```bash
python -m plagiarism_engine.cli corpus --help
```

---

## 3. `pairs` — evaluate on a labeled pair dataset

### Example 1 — synthetic demo dataset, manual thresholds
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/quora/sample_pairs.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --shingle-size 1 \
    --minhash-threshold 0.13 \
    --simhash-threshold 0.46 \
    --output outputs/metrics.csv
```

```bash
cat outputs/metrics.csv
```
Expected: `minhash_lsh` with F1≈0.806, `simhash` with F1≈0.714.

### Example 2 — same dataset, automatic threshold selection instead of guessing
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/quora/sample_pairs.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --shingle-size 1 \
    --sweep \
    --sweep-output outputs/threshold_sweep.csv \
    --output outputs/metrics_swept.csv
```

```bash
cat outputs/metrics_swept.csv
```

```bash
cat outputs/threshold_sweep.csv
```

### Example 3 — only evaluate the first 30 rows
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/quora/sample_pairs.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --limit 30 \
    --shingle-size 1 \
    --sweep \
    --output outputs/metrics_first30.csv
```

### See every available option
```bash
python -m plagiarism_engine.cli pairs --help
```

---

## 4. PAN-PC-11 real dataset (only if you've downloaded it)

These commands assume you already downloaded and extracted PAN-PC-11 to exactly this path:
`data/raw/pan-plagiarism-corpus-2011/external-detection-corpus`
(the same path used in this project's own verified test run). If your extracted folder is somewhere else,
replace that one path in the three commands below — everything else stays the same.

### Step 1 — confirm the real annotation feature name
```bash
python scripts/prepare_pan_pc11_pairs.py \
    --corpus-dir data/raw/pan-plagiarism-corpus-2011/external-detection-corpus \
    --inspect
```
Confirmed on the real 2011 release: the value is `plagiarism` (used in Step 2 below).

### Step 2 — build the labeled pairs CSV
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/processed/pan_pc11_pairs.csv \
    --text-col-a text_a --text-col-b text_b --label-col label \
    --shingle-size 1 --sweep \
    --output outputs/metrics_pan_pc11.csv
```

```bash
head -3 data/processed/pan_pc11_pairs.csv
```

### Step 3 — evaluate (k=1, with automatic threshold selection)
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/processed/pan_pc11_pairs.csv \
    --text-col-a text_a \
    --text-col-b text_b \
    --label-col label \
    --shingle-size 1 \
    --sweep \
    --sweep-output outputs/pan_pc11_threshold_sweep.csv \
    --output outputs/metrics_pan_pc11.csv
```

```bash
cat outputs/metrics_pan_pc11.csv
```
Expected: `minhash_lsh` with F1≈0.920, `simhash` with F1≈0.789.

### Optional — also try the specification-recommended k=3, to see the difference
```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/processed/pan_pc11_pairs.csv \
    --text-col-a text_a \
    --text-col-b text_b \
    --label-col label \
    --shingle-size 3 \
    --sweep \
    --output outputs/metrics_pan_pc11_k3.csv
```

```bash
cat outputs/metrics_pan_pc11_k3.csv
```

### Optional — build a bigger pairs sample (5,000 positive instead of the default 2,000)
```bash
python scripts/prepare_pan_pc11_pairs.py \
    --corpus-dir data/raw/pan-plagiarism-corpus-2011/external-detection-corpus \
    --output data/processed/pan_pc11_pairs_5000.csv \
    --feature-name plagiarism \
    --limit-positive 5000
```

```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/processed/pan_pc11_pairs_5000.csv \
    --text-col-a text_a \
    --text-col-b text_b \
    --label-col label \
    --shingle-size 1 \
    --sweep \
    --output outputs/metrics_pan_pc11_5000.csv
```

### See every available option
```bash
python scripts/prepare_pan_pc11_pairs.py --help
```

---

## 5. Full run, start to finish (everything above, back to back)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
pytest -v
python -m plagiarism_engine.cli compare --file-a data/sample_corpus/doc_01.txt --file-b data/sample_corpus/doc_02.txt
python -m plagiarism_engine.cli corpus --data data/sample_corpus --threshold 0.25 --output outputs/candidates.csv
cat outputs/candidates.csv
python -m plagiarism_engine.cli pairs --pairs data/raw/quora/sample_pairs.csv --text-col-a question1 --text-col-b question2 --label-col is_duplicate --shingle-size 1 --sweep --output outputs/metrics.csv
cat outputs/metrics.csv
python scripts/prepare_pan_pc11_pairs.py --corpus-dir data/raw/pan-plagiarism-corpus-2011/external-detection-corpus --inspect
python scripts/prepare_pan_pc11_pairs.py --corpus-dir data/raw/pan-plagiarism-corpus-2011/external-detection-corpus --output data/processed/pan_pc11_pairs.csv --feature-name plagiarism
python -m plagiarism_engine.cli pairs --pairs data/processed/pan_pc11_pairs.csv --text-col-a text_a --text-col-b text_b --label-col label --shingle-size 1 --sweep --output outputs/metrics_pan_pc11.csv
cat outputs/metrics_pan_pc11.csv
```