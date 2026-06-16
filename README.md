# 🔍 Semantic Plagiarism Engine

A command-line engine for detecting near-duplicate and copied documents using
semantic and structural similarity techniques. Core algorithms are implemented
from scratch for clarity and reproducibility.

## ✨ Features
-  **Shingling + MinHash + LSH** for scalable near-duplicate detection
-  **TF-IDF Weighted SimHash** for semantic similarity
-  Fast candidate retrieval over large document collections
-  Built-in evaluation against ground-truth corpora

## 🚀 Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -e .
```

## 📚 Dataset
PAN Plagiarism Corpus 2011 (PAN-PC-11), available on Zenodo.

