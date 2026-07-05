# `data/raw/`

This directory is where **full-size, third-party datasets** should be
placed if you want to reproduce the large-scale evaluation described in
`docs/project_spec.pdf`. Per the project requirements, large raw datasets
are **not** bundled with this repository -- only a small synthetic demo
file is checked in (see below) so the `pairs` CLI command works out of
the box without any downloads.

## Recommended datasets

| Dataset | Use case | Where to get it |
|---|---|---|
| **PAN-PC-11** (PAN 2011 Plagiarism Corpus) | Primary dataset: manually and automatically obfuscated plagiarism cases | [Zenodo, DOI 10.5281/zenodo.3250095](https://zenodo.org/records/3250095) |
| **Stack Exchange Duplicates** | Alternative: duplicate-marked Q&A post pairs | Hugging Face Datasets Hub |
| **Quora Question Pairs** | Alternative: 400k+ question pairs labeled duplicate / non-duplicate | Hugging Face Datasets Hub / Kaggle |

### Using PAN-PC-11

PAN-PC-11 is **not** a simple two-column CSV like Quora -- it's a real corpus of raw
`suspicious-documentKKKKKK.txt` / `source-documentKKKKKK.txt` files, each suspicious document paired with a
ground-truth `.xml` annotation file that marks *where* a plagiarized passage sits and which source
document (and offset) it came from. **Confirmed real directory layout** (2011 release):

```
pan-plagiarism-corpus-2011/
├── external-detection-corpus/          <-- point --data / --corpus-dir here
│   ├── source-document/
│   │   ├── part1/ ... part23/
│   │   │   ├── source-documentKKKKKK.txt
│   │   │   └── source-documentKKKKKK.xml
│   └── suspicious-document/
│       ├── part1/ ... part23/
│       │   ├── suspicious-documentKKKKKK.txt
│       │   └── suspicious-documentKKKKKK.xml   (meta info if "clean", plagiarism spans if not)
└── intrinsic-detection-corpus/
    └── suspicious-document/
        ├── part1/ ... part10/                  <-- no source documents at all; not usable
                                                      for pairwise duplicate detection
```

Ground-truth annotation schema (per the official PAN task documentation:
<https://pan.webis.de/clef11/pan11-web/external-plagiarism-detection.html>):

```xml
<document reference="suspicious-documentKKKKKK.txt">
    <feature name="plagiarism"
             this_offset="5" this_length="1000"
             source_reference="source-documentABC.txt"
             source_offset="100" source_length="1000" />
</document>
```

**Update: confirmed against a real download.** A real run of `--inspect` against the actual corpus found
this sampled *only* `source-document*.xml` files at first (all with `name="about"`/`name="md5Hash"` --
meta info, not plagiarism cases) because of a now-fixed sampling bug that let a `source-document`-heavy
directory-listing order fill up the `--max-files` cap before ever reaching a `suspicious-document` file.
Despite that, running the full conversion with the documented default (`--feature-name plagiarism`)
**worked correctly** -- `name="plagiarism"` is the real value used in this release's
`suspicious-document*.xml` ground truth. `--inspect` has since been fixed to prioritize
`suspicious-document` files so this is visible directly next time, without relying on this note.

**Important:** always point `--corpus-dir` at `external-detection-corpus` specifically, not the corpus
root and not `intrinsic-detection-corpus` -- the latter has no source documents to pair against at all,
and its `suspicious-document` folder reuses the same numbering scheme as the external corpus's, which the
script will detect and warn about (`WARNING: N duplicate document basenames found...`) but cannot fully
resolve on your behalf.

The script was written directly from PAN's officially documented ground-truth XML schema, tested against
several hand-built fixtures matching that schema and its edge cases (a clean suspicious document with no
plagiarism, an unrecognized feature name, and a colliding document ID across two sub-corpora), and has now
also been confirmed against a real download (22,186 (txt, xml) pairs found; 2,000 positive pairs built
with 0 skipped/0 parse errors at the documented default `--feature-name plagiarism`).

**Download note**: the Zenodo release is split across two `.rar` volumes (~0.85 GB each); if they are a
genuine multi-volume archive, your extraction tool may require both parts to extract anything at all.

There are two ways to point this project at PAN-PC-11 once it's downloaded and extracted, and **neither
one requires changing any file inside `src/plagiarism_engine/`**:

**Option A -- no conversion needed (document-level near-duplicate scan):**

```bash
python -m plagiarism_engine.cli corpus \
    --data data/raw/pan-pc-11/external-detection-corpus/suspicious-document/part1 \
    --threshold 0.25 \
    --output outputs/candidates_pan.csv
```

`load_corpus()` already reads any folder of `*.txt` files, so this works immediately against the raw
extracted documents (start with a single `partN` folder -- each part likely has hundreds of documents,
so `n*(n-1)/2` pairwise comparisons before LSH pruning can get large fast). It reports near-duplicate
*documents*, not per-passage plagiarism, and does not use the ground-truth XML labels at all.

**Option B -- full precision/recall/F1 evaluation against the ground-truth labels:**

Use the conversion script at `scripts/prepare_pan_pc11_pairs.py`, which parses the XML annotations into a
`text_a, text_b, label` CSV that the existing, unmodified `pairs` command can read directly.

```bash
# 1. Discover the real <feature name="..."> value(s) used in your corpus -- do this FIRST:
python scripts/prepare_pan_pc11_pairs.py \
    --corpus-dir data/raw/pan-pc-11/external-detection-corpus --inspect
```

This prints every distinct feature name actually found (with counts and an example), so you don't have to
guess. If it's something other than `"plagiarism"` (e.g. `"artificial-plagiarism"`), no code change is
needed -- just pass it along in the next step:

```bash
# 2. Build the labeled pairs CSV (positive = annotated plagiarism spans,
#    negative = randomly sampled excerpts from unrelated document pairs,
#    excluding any document pair that has a real plagiarism relationship
#    elsewhere in the corpus):
python scripts/prepare_pan_pc11_pairs.py \
    --corpus-dir data/raw/pan-pc-11/external-detection-corpus \
    --output data/processed/pan_pc11_pairs.csv \
    --feature-name plagiarism \
    --limit-positive 2000

# 3. Evaluate. Use --sweep so the decision threshold is chosen to maximize F1
#    on this dataset rather than reusing a threshold tuned for a different
#    kind of text (see "A note on thresholds" below) -- almost free, since
#    the similarity scores are computed once and re-thresholded cheaply:
python -m plagiarism_engine.cli pairs \
    --pairs data/processed/pan_pc11_pairs.csv \
    --text-col-a text_a --text-col-b text_b --label-col label \
    --sweep --sweep-output outputs/pan_pc11_threshold_sweep.csv \
    --output outputs/metrics_pan_pc11.csv
```

### A note on thresholds

**High precision with very low recall is the signature of a too-strict decision threshold, not
necessarily a broken similarity measure.** The default thresholds (`--minhash-threshold 0.5`,
`--simhash-threshold 0.85`) were chosen for paragraph-length, mostly-unobfuscated text; PAN-PC-11
deliberately includes a large share of artificially- and simulated-obfuscated plagiarism cases (by
design -- see the corpus's own documentation), which share much less lexical surface overlap with their
source than a verbatim copy does, similar to the "heavy paraphrase" cases discussed in
`docs/project_spec.pdf`. Always pass `--sweep` (or tune `--minhash-threshold`/`--simhash-threshold`
manually) on any new dataset rather than assuming the defaults transfer.

### Using Quora / Stack Exchange

After downloading, place the pair file(s) under a subfolder here, e.g.
`data/raw/quora/train.csv`, and point the CLI at it:

```bash
python -m plagiarism_engine.cli pairs \
    --pairs data/raw/quora/train.csv \
    --text-col-a question1 \
    --text-col-b question2 \
    --label-col is_duplicate \
    --limit 5000 \
    --output outputs/metrics.csv
```

## `quora/sample_pairs.csv`

This is a **60-row synthetic demo file** (30 duplicate / 30 non-duplicate
question pairs), hand-authored in the Quora Question Pairs schema
(`question1, question2, is_duplicate`) purely so that
`python -m plagiarism_engine.cli pairs` and the metrics reported in
`docs/project_spec.pdf` are reproducible without a network connection.
It is intentionally small and is **not** a substitute for the real
Quora Question Pairs dataset (400,000+ rows) referenced in the project
specification.
