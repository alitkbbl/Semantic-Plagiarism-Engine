# `data/sample_corpus/`

A small, hand-authored demo corpus (10 `.txt` files) used by the
`compare` and `corpus` CLI commands and by `docs/project_spec.pdf`'s
experimental results section. It deliberately covers a range of cases:

| File | Purpose |
|---|---|
| `doc_01.txt` | Source paragraph on renewable energy |
| `doc_02.txt` | Heavily paraphrased near-duplicate of `doc_01.txt` (full rewrite, low lexical overlap) |
| `doc_03.txt` | Verbatim copy of `doc_01.txt` (simulated copy-paste plagiarism) |
| `doc_04.txt` | Unrelated source paragraph on Mars exploration |
| `doc_05.txt` | Heavily paraphrased near-duplicate of `doc_04.txt` |
| `doc_06.txt` | Unrelated topic (cooking pasta), no expected matches |
| `doc_07.txt` | Very short document (edge case: fewer tokens than the shingle size) |
| `doc_08.txt` | Document with emoji, accents and unusual punctuation (edge case) |
| `doc_09.txt` | Empty document (edge case) |
| `doc_10.txt` | Persian-language document (edge case: non-Latin script, Persian stop words) |
| `doc_11.txt` | Lightly-edited near-duplicate of `doc_01.txt` (minor word substitutions, most phrasing intact) |

Running:

```bash
python -m plagiarism_engine.cli corpus \
    --data data/sample_corpus \
    --threshold 0.25 \
    --shingle-size 3 \
    --output outputs/candidates.csv
```

should surface `(doc_01.txt, doc_03.txt)` at similarity 1.0 (verbatim copy),
`(doc_01.txt, doc_11.txt)` at a moderate similarity around 0.48 (light
edit), and should **not** surface `(doc_01.txt, doc_02.txt)` or
`(doc_04.txt, doc_05.txt)` at the default threshold -- both are heavy,
full-sentence paraphrases with very little surviving 3-word overlap,
which is exactly the failure mode discussed in the technical report's
error analysis (shingle/MinHash methods are lexical, not semantic). No
pair involving `doc_06.txt` through `doc_10.txt` should be reported.
