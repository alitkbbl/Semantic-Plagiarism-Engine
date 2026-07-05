"""
dataset.py
==========

Dataset loading and I/O helpers.

Two kinds of input are supported, matching the two "search" style CLI
commands:

* **Folder corpora** (``corpus`` command): a directory of ``*.txt`` files,
  one document per file, e.g. ``data/sample_corpus/``.
* **Labeled pair datasets** (``pairs`` command): a CSV file with two text
  columns and a binary label column, e.g. a Quora-Question-Pairs-style
  file with ``question1, question2, is_duplicate`` columns, or a
  Stack Exchange duplicate-post export, or a PAN-PC-11-derived pair list.

Large raw datasets (PAN-PC-11, Quora Question Pairs, Stack Exchange
Duplicates) are intentionally **not** bundled with this project -- see
``data/raw/README.md`` for download pointers. Only a small synthetic
sample corpus and a small synthetic sample pair file are included so the
CLI is runnable out of the box.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def load_corpus(folder: Path) -> Dict[str, str]:
    """Load every ``*.txt`` file in ``folder`` into a {filename: text} dict.

    Files are read as UTF-8 with errors replaced rather than raised, so a
    single malformed file does not abort processing of an entire corpus.
    Sub-directories are not recursed into.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Corpus folder not found: {folder}")

    documents: Dict[str, str] = {}
    for path in sorted(folder.glob("*.txt")):
        try:
            documents[path.name] = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:  # pragma: no cover - defensive
            raise OSError(f"Could not read {path}: {exc}") from exc
    return documents


def load_pairs_csv(
    path: Path,
    text_col_a: str,
    text_col_b: str,
    label_col: Optional[str] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Load a labeled (or unlabeled) pair dataset from CSV.

    Rows with missing text in either column are dropped. If ``limit`` is
    given, only the first ``limit`` valid rows are returned (useful for
    quickly evaluating on a subset of a large dataset such as Quora
    Question Pairs, which has 400k+ rows).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pairs file not found: {path}")

    usecols = [text_col_a, text_col_b] + ([label_col] if label_col else [])
    df = pd.read_csv(path, usecols=usecols)
    df = df.dropna(subset=[text_col_a, text_col_b])
    df[text_col_a] = df[text_col_a].astype(str)
    df[text_col_b] = df[text_col_b].astype(str)

    if label_col:
        df = df.dropna(subset=[label_col])
        df[label_col] = df[label_col].astype(int)

    if limit is not None:
        df = df.head(limit)

    return df.reset_index(drop=True)


def save_json(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False, default=str)


def save_rows_csv(rows: List[dict], path: Path, fieldnames: Optional[List[str]] = None) -> None:
    """Write a list of flat dicts to CSV, creating parent directories.

    If the list of rows is empty, an (empty) file with just the header
    (if ``fieldnames`` is given) or a fully empty file is still written,
    so downstream tooling can rely on the output path always existing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []

    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
