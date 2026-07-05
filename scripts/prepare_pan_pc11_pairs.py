#!/usr/bin/env python3
"""
prepare_pan_pc11_pairs.py
==========================

One-time, offline data-prep utility: converts a locally-extracted copy of
the real PAN-PC-11 corpus (raw suspicious-document / source-document
``*.txt`` files plus ground-truth ``*.xml`` annotation files) into a
labeled pair CSV compatible with the existing, UNMODIFIED ``pairs`` CLI
command:

    python -m plagiarism_engine.cli pairs \\
        --pairs <output.csv> --text-col-a text_a --text-col-b text_b \\
        --label-col label --output outputs/metrics_pan_pc11.csv

This script is intentionally kept OUTSIDE ``src/plagiarism_engine/`` --
it is a one-time offline conversion step you run locally after
downloading and extracting PAN-PC-11 from
https://zenodo.org/records/3250095. Nothing in the plagiarism_engine
package itself needs to change to consume its output: the resulting CSV
is read by the existing ``plagiarism_engine.dataset.load_pairs_csv()``
exactly like any other pairs file (e.g. the synthetic demo file at
``data/raw/quora/sample_pairs.csv``).

Confirmed real directory layout (2011 release):

    pan-plagiarism-corpus-2011/
        external-detection-corpus/       <-- point --corpus-dir here
            source-document/
                part1/ ... part23/
                    source-documentKKKKKK.txt
                    source-documentKKKKKK.xml
            suspicious-document/
                part1/ ... part23/
                    suspicious-documentKKKKKK.txt
                    suspicious-documentKKKKKK.xml
        intrinsic-detection-corpus/
            suspicious-document/         <-- NOT used by this script:
                part1/ ... part10/       no source documents at all, so
                                          there is nothing to pair against.

Always point ``--corpus-dir`` at ``external-detection-corpus`` specifically
(not the corpus root, and not just one of the two document-type folders):
this script needs to see *both* ``source-document/`` and
``suspicious-document/`` at once to resolve a case's ``source_reference``,
and skipping ``intrinsic-detection-corpus`` avoids any risk of document
ID numbering colliding between the two independent sub-corpora.

Ground-truth XML schema (per the official PAN task documentation --
https://pan.webis.de/clef11/pan11-web/external-plagiarism-detection.html):

    <document reference="suspicious-documentXYZ.txt">
        <feature name="plagiarism"
                 this_offset="5" this_length="1000"
                 source_reference="source-documentABC.txt"
                 source_offset="100" source_length="1000" />
        ...
    </document>

IMPORTANT: the exact string used for ``name="..."`` on real plagiarism-case
features has not been independently confirmed against an actual PAN-PC-11
file (this script was written without network/dataset access). Run with
``--inspect`` FIRST -- it scans your real corpus and prints every distinct
``<feature name="...">`` value actually found, with counts, so you can see
the true value directly instead of trusting this docstring. If it turns out
to be something other than ``"plagiarism"``, pass
``--feature-name <the real value>`` -- no code changes needed.

Usage
-----
    # 1. Discover the real feature name(s) used in your corpus:
    python scripts/prepare_pan_pc11_pairs.py \\
        --corpus-dir data/raw/pan-pc-11/external-detection-corpus --inspect

    # 2. Generate the full labeled pairs CSV (add --feature-name if step 1
    #    showed a different value than "plagiarism"):
    python scripts/prepare_pan_pc11_pairs.py \\
        --corpus-dir data/raw/pan-pc-11/external-detection-corpus \\
        --output data/processed/pan_pc11_pairs.csv \\
        --limit-positive 2000

    # 3. Evaluate with the existing CLI (unchanged):
    python -m plagiarism_engine.cli pairs \\
        --pairs data/processed/pan_pc11_pairs.csv \\
        --text-col-a text_a --text-col-b text_b --label-col label \\
        --output outputs/metrics_pan_pc11.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def find_txt_xml_pairs(root: Path):
    """Recursively find every (txt, xml) file pair under ``root``, matching
    a text file only to an XML file with the same stem *in the same
    directory* (never by stem alone across the whole tree) -- this matters
    because PAN-PC-11's external- and intrinsic-detection sub-corpora reuse
    the same document numbering independently, so e.g.
    ``suspicious-document00001`` exists under both; matching by stem alone
    across the whole tree could silently pair a document with the wrong
    sub-corpus's annotation file."""
    pairs = []
    for txt_path in root.rglob("*.txt"):
        xml_path = txt_path.with_suffix(".xml")
        if xml_path.exists():
            pairs.append((txt_path, xml_path))
    return pairs


def index_documents(root: Path, warn_collisions: bool = True):
    """Map every *.txt filename (basename) found under root to its full
    path, so a ``source_reference`` can be resolved regardless of which
    subdirectory it actually lives in.

    If the same basename appears more than once under ``root`` (e.g. because
    --corpus-dir was pointed at a level that mixes two independently
    ID-numbered sub-corpora), the first one found wins and a warning is
    printed -- this should not happen if --corpus-dir is pointed at
    external-detection-corpus specifically, per the module docstring.
    """
    index: dict = {}
    collisions = 0
    for txt_path in root.rglob("*.txt"):
        if txt_path.name in index:
            collisions += 1
            continue
        index[txt_path.name] = txt_path
    if warn_collisions and collisions:
        print(f"WARNING: {collisions} duplicate document basenames found under {root}; "
              f"the first occurrence of each was kept. This usually means --corpus-dir "
              f"is pointed too high up the tree (e.g. at the corpus root instead of "
              f"external-detection-corpus/) -- see the module docstring.", file=sys.stderr)
    return index


def discover_feature_names(xml_path: Path):
    """Return a Counter of every distinct <feature name="..."> value found
    in one annotation file, regardless of what that value is. Used by
    --inspect to reveal the *real* name in use without guessing."""
    from collections import Counter
    tree = ET.parse(xml_path)
    root_el = tree.getroot()
    counts = Counter()
    for feat in root_el.findall(".//feature"):
        counts[feat.get("name")] += 1
    return counts


def parse_plagiarism_features(xml_path: Path, feature_name: str = "plagiarism"):
    """Parse the ground-truth <feature name="{feature_name}" .../> entries
    from one suspicious-document's annotation file. ``feature_name``
    defaults to "plagiarism" per the officially documented schema, but is
    configurable via --feature-name in case a given release uses a
    different value (confirm the real value with --inspect first)."""
    tree = ET.parse(xml_path)
    root_el = tree.getroot()
    features = []
    for feat in root_el.findall(".//feature"):
        if feat.get("name") != feature_name:
            continue
        this_offset, this_length = feat.get("this_offset"), feat.get("this_length")
        if this_offset is None or this_length is None:
            continue
        try:
            this_offset, this_length = int(this_offset), int(this_length)
        except ValueError:
            continue
        source_offset = feat.get("source_offset")
        source_length = feat.get("source_length")
        features.append({
            "this_offset": this_offset,
            "this_length": this_length,
            "source_reference": feat.get("source_reference"),
            "source_offset": int(source_offset) if source_offset is not None else None,
            "source_length": int(source_length) if source_length is not None else None,
        })
    return features


def read_span(path: Path, offset: int, length: int, cache: dict) -> str:
    text = cache.get(path)
    if text is None:
        text = path.read_text(encoding="utf-8", errors="replace")
        cache[path] = text
    return text[offset: offset + length]


def build_positive_pairs(root: Path, limit, min_chars: int, feature_name: str = "plagiarism"):
    doc_index = index_documents(root)
    txt_xml_pairs = find_txt_xml_pairs(root)
    text_cache: dict = {}
    rows = []
    skipped_missing_source = 0
    skipped_too_short = 0
    parse_errors = 0

    for txt_path, xml_path in txt_xml_pairs:
        try:
            features = parse_plagiarism_features(xml_path, feature_name=feature_name)
        except ET.ParseError:
            parse_errors += 1
            continue

        for feat in features:
            if feat["source_reference"] is None or feat["source_offset"] is None:
                continue  # no source identified for this case (allowed by the schema)
            source_path = doc_index.get(feat["source_reference"])
            if source_path is None:
                skipped_missing_source += 1
                continue

            span_a = read_span(txt_path, feat["this_offset"], feat["this_length"], text_cache).strip()
            span_b = read_span(source_path, feat["source_offset"], feat["source_length"], text_cache).strip()
            if len(span_a) < min_chars or len(span_b) < min_chars:
                skipped_too_short += 1
                continue

            rows.append({
                "text_a": span_a, "text_b": span_b, "label": 1,
                "doc_a": txt_path.name, "doc_b": source_path.name,
            })
            if limit and len(rows) >= limit:
                return rows, {
                    "skipped_missing_source": skipped_missing_source,
                    "skipped_too_short": skipped_too_short,
                    "parse_errors": parse_errors,
                }

    return rows, {
        "skipped_missing_source": skipped_missing_source,
        "skipped_too_short": skipped_too_short,
        "parse_errors": parse_errors,
    }


def _excerpt(text: str, rng: random.Random, target_len: int = 1000) -> str:
    """Take a bounded-length excerpt so negative pairs are on a similar
    length scale to the positive plagiarism spans, rather than comparing
    two entire documents (which would trivially look dissimilar)."""
    text = text.strip()
    if len(text) <= target_len:
        return text
    start = rng.randint(0, len(text) - target_len)
    return text[start:start + target_len].strip()


def build_negative_pairs(root: Path, num_negatives: int, min_chars: int,
                          excluded_doc_pairs: set | None = None, seed: int = 42):
    doc_index = index_documents(root)
    doc_paths = list(doc_index.values())
    if len(doc_paths) < 2:
        return []
    excluded_doc_pairs = excluded_doc_pairs or set()

    rng = random.Random(seed)
    text_cache: dict = {}
    rows = []
    attempts, max_attempts = 0, num_negatives * 20

    while len(rows) < num_negatives and attempts < max_attempts:
        attempts += 1
        a, b = rng.sample(doc_paths, 2)
        if frozenset((a.name, b.name)) in excluded_doc_pairs:
            # These two documents have at least one known plagiarism
            # relationship elsewhere in the corpus (per the ground-truth
            # XML); skip them so we never mislabel a genuinely-related
            # document pair as a negative just because we sampled a
            # different, non-overlapping excerpt from each.
            continue
        for p in (a, b):
            if p not in text_cache:
                text_cache[p] = p.read_text(encoding="utf-8", errors="replace")

        span_a = _excerpt(text_cache[a], rng)
        span_b = _excerpt(text_cache[b], rng)
        if len(span_a) < min_chars or len(span_b) < min_chars:
            continue

        rows.append({
            "text_a": span_a, "text_b": span_b, "label": 0,
            "doc_a": a.name, "doc_b": b.name,
        })

    return rows


def run_inspect(corpus_dir: Path, max_files: int = 3000) -> None:
    from collections import Counter

    txt_xml_pairs = find_txt_xml_pairs(corpus_dir)
    print(f"Found {len(txt_xml_pairs)} (txt, xml) file pairs under {corpus_dir}\n")
    if not txt_xml_pairs:
        print("No matching pairs found. Check --corpus-dir points at a folder that contains")
        print("both suspicious-documentNNNNNN.txt and suspicious-documentNNNNNN.xml files")
        print("(e.g. .../external-detection-corpus).")
        return

    # Plagiarism cases are only ever annotated in suspicious-document*.xml
    # files -- source-document*.xml files only carry meta information
    # (e.g. "about"/"md5Hash" features). Sort those first so a --max-files
    # cap can't accidentally sample only source-document files and report
    # a false "no plagiarism feature found" simply because it never looked
    # at a suspicious document at all.
    txt_xml_pairs = sorted(
        txt_xml_pairs, key=lambda pair: not pair[0].name.startswith("suspicious")
    )
    n_suspicious = sum(1 for txt_path, _ in txt_xml_pairs if txt_path.name.startswith("suspicious"))
    n_source = len(txt_xml_pairs) - n_suspicious
    print(f"({n_suspicious} suspicious-document / {n_source} source-document pairs total; "
          f"prioritizing suspicious-document files in the sample below, since that's where "
          f"plagiarism annotations live.)\n")

    sample = txt_xml_pairs[:max_files]
    truncated = len(txt_xml_pairs) > max_files

    print(f"Scanning {len(sample)} annotation file(s) for every <feature name=\"...\"> value "
          f"actually in use{' (capped by --max-files)' if truncated else ''}...\n")

    name_counts: Counter = Counter()
    parse_errors = 0
    example_by_name: dict = {}
    for _, xml_path in sample:
        try:
            counts = discover_feature_names(xml_path)
        except ET.ParseError:
            parse_errors += 1
            continue
        name_counts.update(counts)
        for name in counts:
            example_by_name.setdefault(name, xml_path)

    if not name_counts:
        print("No <feature> elements found at all in the sampled files (they may all be")
        print("'clean' suspicious documents with no injected plagiarism, or source-document")
        print("meta-info files). Try a larger --max-files, or open a suspicious-document*.xml")
        print("file directly to check its structure.")
    else:
        print("Distinct <feature name=\"...\"> values found, with counts:")
        for name, count in name_counts.most_common():
            print(f"  {name!r:35s} {count:>8d} occurrences   (example: {example_by_name[name].name})")

        print("\nIf a value above clearly corresponds to injected plagiarism cases (it should")
        print("have this_offset / this_length / source_reference / source_offset / source_length")
        print("attributes -- shown below), pass it as --feature-name if it isn't \"plagiarism\".")
        print("Values like 'about' or 'md5Hash' are source-document META INFO, not plagiarism")
        print("cases -- ignore those.")

        plagiarism_like = [n for n in name_counts if n and "plagiar" in n.lower()]
        for name in (plagiarism_like or list(name_counts)[:1]):
            example_xml = example_by_name[name]
            print(f"\nExample <feature name={name!r}> entries from {example_xml}:")
            for feat in parse_plagiarism_features(example_xml, feature_name=name)[:3]:
                print(f"    {feat}")

    if parse_errors:
        print(f"\n{parse_errors} file(s) failed to parse as XML.")

    print("\nOnce you've confirmed the right --feature-name, re-run without --inspect to build the full CSV.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--corpus-dir", required=True, type=Path,
                         help="Root folder of the extracted PAN-PC-11 corpus (any nesting depth).")
    parser.add_argument("--output", type=Path,
                         help="Where to write the pairs CSV (required unless --inspect).")
    parser.add_argument("--limit-positive", type=int, default=2000,
                         help="Cap on the number of positive (plagiarism) pairs to extract.")
    parser.add_argument("--num-negatives", type=int, default=None,
                         help="Defaults to the same count as positive pairs found (balanced dataset).")
    parser.add_argument("--min-chars", type=int, default=20,
                         help="Discard spans shorter than this many characters.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-name", default="plagiarism",
                         help="The <feature name=\"...\"> value that marks a plagiarism case in the "
                              "ground-truth XML. Default 'plagiarism' per the documented schema -- "
                              "run --inspect first to confirm the real value in your corpus.")
    parser.add_argument("--inspect", action="store_true",
                         help="Scan the corpus and report every real <feature name=\"...\"> value "
                              "found, then exit. Use this FIRST, before generating the full CSV.")
    parser.add_argument("--max-files", type=int, default=3000,
                         help="Cap on how many annotation files --inspect scans (for speed on huge corpora).")
    args = parser.parse_args()

    if not args.corpus_dir.is_dir():
        sys.exit(f"Not a directory: {args.corpus_dir}")

    if args.inspect:
        run_inspect(args.corpus_dir, max_files=args.max_files)
        return

    if args.output is None:
        sys.exit("--output is required unless --inspect is given.")

    print(f"Scanning {args.corpus_dir} for annotated plagiarism pairs "
          f"(feature name={args.feature_name!r})...")
    positive_rows, stats = build_positive_pairs(
        args.corpus_dir, args.limit_positive, args.min_chars, feature_name=args.feature_name,
    )
    print(f"Built {len(positive_rows)} positive pairs.")
    print(f"  Skipped (source document not found in corpus tree): {stats['skipped_missing_source']}")
    print(f"  Skipped (span shorter than --min-chars): {stats['skipped_too_short']}")
    print(f"  XML files that failed to parse: {stats['parse_errors']}")

    if not positive_rows:
        sys.exit(
            "\nNo positive pairs were extracted. Re-run with --inspect to check whether "
            "the XML schema in your corpus matches what this script expects."
        )

    num_negatives = args.num_negatives if args.num_negatives is not None else len(positive_rows)
    excluded_doc_pairs = {frozenset((r["doc_a"], r["doc_b"])) for r in positive_rows}
    negative_rows = build_negative_pairs(
        args.corpus_dir, num_negatives, args.min_chars,
        excluded_doc_pairs=excluded_doc_pairs, seed=args.seed,
    )
    print(f"Built {len(negative_rows)} negative pairs (randomly sampled document excerpts, "
          f"excluding {len(excluded_doc_pairs)} document pairs with a known plagiarism relationship).")

    all_rows = positive_rows + negative_rows
    random.Random(args.seed).shuffle(all_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text_a", "text_b", "label", "doc_a", "doc_b"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote {len(all_rows)} total pairs ({len(positive_rows)} positive / "
          f"{len(negative_rows)} negative) to {args.output}")
    print("\nNext step -- evaluate with the existing, unmodified `pairs` command:")
    print(f"  python -m plagiarism_engine.cli pairs \\")
    print(f"      --pairs {args.output} \\")
    print(f"      --text-col-a text_a --text-col-b text_b --label-col label \\")
    print(f"      --output outputs/metrics_pan_pc11.csv")


if __name__ == "__main__":
    main()
