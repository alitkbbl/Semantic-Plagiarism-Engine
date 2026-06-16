#!/usr/bin/env python3
"""Scaffold the repository structure for semantic-plagiarism-engine."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# دایرکتوری‌ها
DIRS = [
    ".github/workflows",
    "docs",
    "data/sample_corpus/raw",
    "data/sample_corpus/processed",
    "src/plagiarism_engine",
    "notebooks",
    "tests",
    "outputs",
]

# فایل‌های خالی
FILES = [
    "requirements.txt",
    "pyproject.toml",
    ".gitignore",
    ".github/workflows/tests.yml",
    "docs/project_spec.tex",
    # docs/project_spec.pdf را به‌صورت دستی اضافه کن (باینری)
    "src/plagiarism_engine/__init__.py",
    "src/plagiarism_engine/preprocessing.py",
    "src/plagiarism_engine/minhash.py",
    "src/plagiarism_engine/lsh.py",
    "src/plagiarism_engine/simhash.py",
    "src/plagiarism_engine/dataset.py",
    "src/plagiarism_engine/evaluation.py",
    "src/plagiarism_engine/cli.py",
    "notebooks/exploration.ipynb",
    "tests/test_engine.py",
    "outputs/metrics.csv",
    "outputs/candidates.csv",
]

# پوشه‌هایی که توسط git نادیده گرفته می‌شوند → با .gitkeep ردگیری ساختار
GITKEEP_DIRS = [
    "data/sample_corpus/raw",
    "data/sample_corpus/processed",
]


def main() -> None:
    for d in DIRS:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
        print(f"dir   {d}/")

    for f in FILES:
        path = ROOT / f
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.touch()
            print(f"file  {f}")
        else:
            print(f"skip  {f} (exists)")

    for d in GITKEEP_DIRS:
        keep = ROOT / d / ".gitkeep"
        if not keep.exists():
            keep.touch()
            print(f"keep  {d}/.gitkeep")


if __name__ == "__main__":
    main()
