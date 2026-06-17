# src/cli.py
"""
Command-line interface for the Semantic Plagiarism Detection Engine.

This CLI provides three main commands as required by DM_P3_Guide.pdf (Page 6):
1. compare: Compare two text files
2. corpus: Search for similar documents in a corpus
3. pairs: Evaluate on labeled pair dataset

Usage:
    python -m plagiarism_engine.cli compare --file-a <path> --file-b <path> [--output <path>]
    python -m plagiarism_engine.cli corpus --data <dir> [--threshold <float>] [--shingle-size <int>] [--output <path>]
    python -m plagiarism_engine.cli pairs --pairs <csv> --text-col-a <col> --text-col-b <col> --label-col <col> [--limit <int>] [--output <path>]
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv

import numpy as np
import pandas as pd

from .preprocessing import WordShingles, TextPreprocessor
from .minhash import MinHash, MinHashIndex
from .lsh import LSH, choose_lsh_parameters
from .simhash import SimHash, SimHashIndex


class PlagiarismCLI:
    """
    Main CLI controller for plagiarism detection engine.
    """
    
    def __init__(self):
        """Initialize CLI with default configurations."""
        self.shingle_size = 3  # Default as per guide (3-5 recommended)
        self.minhash_perms = 128  # Default signature length
        self.simhash_use_tfidf = True
        self.lsh_threshold = 0.5
        
    def setup_argparse(self) -> argparse.ArgumentParser:
        """
        Setup argument parser with all three required commands.
        
        Returns:
            Configured ArgumentParser
        """
        parser = argparse.ArgumentParser(
            prog="plagiarism_engine",
            description="Semantic Plagiarism Detection Engine - DM Course Project",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # Command 1: compare two files
        compare_parser = subparsers.add_parser(
            "compare",
            help="Compare two text files for similarity"
        )
        compare_parser.add_argument(
            "--file-a",
            required=True,
            type=str,
            help="Path to first text file"
        )
        compare_parser.add_argument(
            "--file-b",
            required=True,
            type=str,
            help="Path to second text file"
        )
        compare_parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Output JSON file path (default: print to stdout)"
        )
        compare_parser.add_argument(
            "--shingle-size",
            type=int,
            default=3,
            help="K-shingle size (default: 3)"
        )
        compare_parser.add_argument(
            "--minhash-perms",
            type=int,
            default=128,
            help="MinHash signature length (default: 128)"
        )
        
        # Command 2: search corpus
        corpus_parser = subparsers.add_parser(
            "corpus",
            help="Search for similar documents in a corpus directory"
        )
        corpus_parser.add_argument(
            "--data",
            required=True,
            type=str,
            help="Path to corpus directory containing text files"
        )
        corpus_parser.add_argument(
            "--threshold",
            type=float,
            default=0.25,
            help="Similarity threshold for candidate pairs (default: 0.25)"
        )
        corpus_parser.add_argument(
            "--shingle-size",
            type=int,
            default=3,
            help="K-shingle size (default: 3)"
        )
        corpus_parser.add_argument(
            "--minhash-perms",
            type=int,
            default=128,
            help="MinHash signature length (default: 128)"
        )
        corpus_parser.add_argument(
            "--use-lsh",
            action="store_true",
            help="Use LSH for candidate generation (faster for large corpora)"
        )
        corpus_parser.add_argument(
            "--lsh-bands",
            type=int,
            default=None,
            help="Number of LSH bands (auto-selected if not specified)"
        )
        corpus_parser.add_argument(
            "--output",
            type=str,
            default="outputs/candidates.csv",
            help="Output CSV file path (default: outputs/candidates.csv)"
        )
        corpus_parser.add_argument(
            "--method",
            type=str,
            choices=["minhash", "simhash", "both"],
            default="both",
            help="Similarity method to use (default: both)"
        )
        
        # Command 3: evaluate on labeled pairs
        pairs_parser = subparsers.add_parser(
            "pairs",
            help="Evaluate on labeled pair dataset"
        )
        pairs_parser.add_argument(
            "--pairs",
            required=True,
            type=str,
            help="Path to labeled pairs CSV file"
        )
        pairs_parser.add_argument(
            "--text-col-a",
            required=True,
            type=str,
            help="Column name for first text"
        )
        pairs_parser.add_argument(
            "--text-col-b",
            required=True,
            type=str,
            help="Column name for second text"
        )
        pairs_parser.add_argument(
            "--label-col",
            required=True,
            type=str,
            help="Column name for ground truth label (0/1)"
        )
        pairs_parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Limit number of pairs to evaluate (for testing)"
        )
        pairs_parser.add_argument(
            "--shingle-size",
            type=int,
            default=3,
            help="K-shingle size (default: 3)"
        )
        pairs_parser.add_argument(
            "--minhash-perms",
            type=int,
            default=128,
            help="MinHash signature length (default: 128)"
        )
        pairs_parser.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Similarity threshold for classification (default: 0.5)"
        )
        pairs_parser.add_argument(
            "--output",
            type=str,
            default="outputs/metrics.csv",
            help="Output metrics CSV file path (default: outputs/metrics.csv)"
        )
        pairs_parser.add_argument(
            "--method",
            type=str,
            choices=["jaccard", "minhash", "simhash", "all"],
            default="all",
            help="Evaluation method (default: all)"
        )
        
        return parser
    
    def read_file(self, filepath: str) -> str:
        """
        Read text file content.
        
        Args:
            filepath: Path to text file
            
        Returns:
            File content as string
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading file {filepath}: {e}", file=sys.stderr)
            sys.exit(1)
    
    def write_json(self, data: Dict, output_path: Optional[str] = None) -> None:
        """
        Write data to JSON file or stdout.
        
        Args:
            data: Dictionary to write
            output_path: Output file path (None = stdout)
        """
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(json_str)
            
            print(f"Results written to: {output_path}")
        else:
            print(json_str)
    
    def write_csv(self, data: List[Dict], output_path: str) -> None:
        """
        Write data to CSV file.
        
        Args:
            data: List of dictionaries (rows)
            output_path: Output CSV file path
        """
        if not data:
            print("Warning: No data to write", file=sys.stderr)
            return
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        fieldnames = list(data[0].keys())
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"Results written to: {output_path}")
    
    def command_compare(self, args) -> None:
        """
        Command 1: Compare two text files.
        
        Computes similarity using multiple methods:
        - Jaccard similarity (exact)
        - MinHash + LSH approximation
        - SimHash with Hamming distance
        
        Args:
            args: Parsed command-line arguments
        """
        print("=" * 60)
        print("COMMAND: Compare Two Files")
        print("=" * 60)
        
        # Read files
        print(f"\nReading file A: {args.file_a}")
        text_a = self.read_file(args.file_a)
        
        print(f"Reading file B: {args.file_b}")
        text_b = self.read_file(args.file_b)
        
        print(f"\nFile A length: {len(text_a)} characters")
        print(f"File B length: {len(text_b)} characters")
        
        # Initialize components
        shingler = WordShingles(k=args.shingle_size)
        minhash = MinHash(num_permutations=args.minhash_perms)
        simhash = SimHash(use_tfidf=False)  # No corpus for TF-IDF in pairwise
        
        start_time = time.time()
        
        # Method 1: Exact Jaccard similarity
        print("\n[1/3] Computing exact Jaccard similarity...")
        shingles_a = shingler.generate_shingles(text_a)
        shingles_b = shingler.generate_shingles(text_b)
        jaccard_sim = shingler.jaccard_similarity(shingles_a, shingles_b)
        
        print(f"  Shingles A: {len(shingles_a)}")
        print(f"  Shingles B: {len(shingles_b)}")
        print(f"  Jaccard similarity: {jaccard_sim:.4f}")
        
        # Method 2: MinHash approximation
        print("\n[2/3] Computing MinHash signatures...")
        sig_a = minhash.compute_signature(shingles_a)
        sig_b = minhash.compute_signature(shingles_b)
        minhash_sim = minhash.estimate_similarity(sig_a, sig_b)
        minhash_error = abs(minhash_sim - jaccard_sim)
        
        print(f"  MinHash similarity: {minhash_sim:.4f}")
        print(f"  Approximation error: {minhash_error:.4f}")
        
        # Method 3: SimHash
        print("\n[3/3] Computing SimHash fingerprints...")
        fp_a = simhash.compute_fingerprint(text_a)
        fp_b = simhash.compute_fingerprint(text_b)
        hamming_dist = simhash.hamming_distance(fp_a, fp_b)
        simhash_sim = simhash.similarity(fp_a, fp_b)
        
        print(f"  Hamming distance: {hamming_dist}/64 bits")
        print(f"  SimHash similarity: {simhash_sim:.4f}")
        
        elapsed_time = time.time() - start_time
        
        # Prepare results
        results = {
            "file_a": args.file_a,
            "file_b": args.file_b,
            "parameters": {
                "shingle_size": args.shingle_size,
                "minhash_permutations": args.minhash_perms
            },
            "statistics": {
                "text_a_length": len(text_a),
                "text_b_length": len(text_b),
                "shingles_a_count": len(shingles_a),
                "shingles_b_count": len(shingles_b),
                "shingles_intersection": len(shingles_a & shingles_b),
                "shingles_union": len(shingles_a | shingles_b)
            },
            "similarities": {
                "jaccard": round(jaccard_sim, 6),
                "minhash": round(minhash_sim, 6),
                "minhash_error": round(minhash_error, 6),
                "simhash": round(simhash_sim, 6),
                "simhash_hamming_distance": hamming_dist
            },
            "execution_time_seconds": round(elapsed_time, 3)
        }
        
        # Output results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"Jaccard similarity:  {jaccard_sim:.4f}")
        print(f"MinHash similarity:  {minhash_sim:.4f}")
        print(f"SimHash similarity:  {simhash_sim:.4f}")
        print(f"Execution time:      {elapsed_time:.3f}s")
        print("=" * 60)
        
        self.write_json(results, args.output)
    
    def load_corpus(self, corpus_dir: str) -> Dict[str, str]:
        """
        Load all text files from a corpus directory.
        
        Args:
            corpus_dir: Path to corpus directory
            
        Returns:
            Dictionary mapping filename to text content
        """
        corpus_path = Path(corpus_dir)
        
        if not corpus_path.exists():
            print(f"Error: Corpus directory not found: {corpus_dir}", file=sys.stderr)
            sys.exit(1)
        
        if not corpus_path.is_dir():
            print(f"Error: Not a directory: {corpus_dir}", file=sys.stderr)
            sys.exit(1)
        
        documents = {}
        text_extensions = {'.txt', '.text', '.doc'}
        
        for filepath in corpus_path.rglob('*'):
            if filepath.is_file() and filepath.suffix.lower() in text_extensions:
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Use relative path as doc_id
                    doc_id = str(filepath.relative_to(corpus_path))
                    documents[doc_id] = content
                except Exception as e:
                    print(f"Warning: Could not read {filepath}: {e}", file=sys.stderr)
        
        if not documents:
            print(f"Error: No text files found in {corpus_dir}", file=sys.stderr)
            sys.exit(1)
        
        return documents
    
    def command_corpus(self, args) -> None:
        """
        Command 2: Search for similar documents in a corpus.
        
        Uses LSH + MinHash and/or SimHash to find candidate pairs
        above the similarity threshold.
        
        Args:
            args: Parsed command-line arguments
        """
        print("=" * 60)
        print("COMMAND: Corpus Similarity Search")
        print("=" * 60)
        
        # Load corpus
        print(f"\nLoading corpus from: {args.data}")
        documents = self.load_corpus(args.data)
        print(f"Loaded {len(documents)} documents")
        
        # Initialize components
        shingler = WordShingles(k=args.shingle_size)
        
        results = []
        
        # Method 1: MinHash + LSH
        if args.method in ["minhash", "both"]:
            print("\n" + "=" * 60)
            print("METHOD 1: MinHash + LSH")
            print("=" * 60)
            
            start_time = time.time()
            
            # Generate shingles
            print("\n[1/4] Generating shingles...")
            doc_shingles = {}
            for doc_id, text in documents.items():
                doc_shingles[doc_id] = shingler.generate_shingles(text)
            
            # Compute MinHash signatures
            print("[2/4] Computing MinHash signatures...")
            minhash = MinHash(num_permutations=args.minhash_perms)
            signatures = {}
            for doc_id, shingles in doc_shingles.items():
                signatures[doc_id] = minhash.compute_signature(shingles)
            
            # Setup LSH
            if args.use_lsh:
                print("[3/4] Building LSH index...")
                if args.lsh_bands:
                    bands = args.lsh_bands
                    rows = args.minhash_perms // bands
                else:
                    bands, rows = choose_lsh_parameters(
                        args.minhash_perms,
                        target_threshold=args.threshold
                    )
                
                print(f"  LSH parameters: {bands} bands × {rows} rows")
                
                lsh = LSH(num_bands=bands, rows_per_band=rows)
                
                for doc_id, sig in signatures.items():
                    lsh.add_document(doc_id, sig)
                
                # Get candidate pairs
                print("[4/4] Finding candidate pairs...")
                candidate_pairs = lsh.get_candidate_pairs()
                
                # Get LSH statistics
                lsh_stats = lsh.get_statistics()
                print(f"\n  Total documents: {lsh_stats['num_documents']}")
                print(f"  Candidate pairs: {lsh_stats['candidate_pairs']}")
                print(f"  All possible pairs: {lsh_stats['all_possible_pairs']}")
                print(f"  Reduction ratio: {lsh_stats['reduction_ratio']:.2%}")
                
            else:
                # All-pairs comparison
                print("[3/4] Performing all-pairs comparison...")
                doc_ids = list(signatures.keys())
                candidate_pairs = set()
                
                for i in range(len(doc_ids)):
                    for j in range(i + 1, len(doc_ids)):
                        candidate_pairs.add((doc_ids[i], doc_ids[j]))
                
                print(f"  Comparing {len(candidate_pairs)} pairs")
            
            # Verify candidates
            print("\nVerifying candidates with MinHash...")
            minhash_results = []
            
            for doc_a, doc_b in candidate_pairs:
                sim = minhash.estimate_similarity(signatures[doc_a], signatures[doc_b])
                
                if sim >= args.threshold:
                    minhash_results.append({
                        "doc_a": doc_a,
                        "doc_b": doc_b,
                        "similarity": round(sim, 6),
                        "method": "minhash"
                    })
            
            minhash_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            elapsed_minhash = time.time() - start_time
            
            print(f"\nMinHash results: {len(minhash_results)} pairs above threshold")
            print(f"Execution time: {elapsed_minhash:.3f}s")
            
            results.extend(minhash_results)
        
        # Method 2: SimHash
        if args.method in ["simhash", "both"]:
            print("\n" + "=" * 60)
            print("METHOD 2: SimHash")
            print("=" * 60)
            
            start_time = time.time()
            
            # Fit SimHash on corpus for TF-IDF
            print("\n[1/3] Fitting SimHash with TF-IDF...")
            simhash = SimHash(use_tfidf=True)
            simhash.fit(list(documents.values()))
            
            # Compute fingerprints
            print("[2/3] Computing SimHash fingerprints...")
            fingerprints = {}
            for doc_id, text in documents.items():
                fingerprints[doc_id] = simhash.compute_fingerprint(text)
            
            # Find similar pairs
            print("[3/3] Finding similar pairs...")
            doc_ids = list(fingerprints.keys())
            simhash_results = []
            
            # Convert threshold to Hamming distance
            max_hamming = int((1.0 - args.threshold) * SimHash.HASH_BITS)
            
            for i in range(len(doc_ids)):
                for j in range(i + 1, len(doc_ids)):
                    doc_a = doc_ids[i]
                    doc_b = doc_ids[j]
                    
                    hamming = simhash.hamming_distance(
                        fingerprints[doc_a],
                        fingerprints[doc_b]
                    )
                    sim = simhash.similarity(
                        fingerprints[doc_a],
                        fingerprints[doc_b]
                    )
                    
                    if sim >= args.threshold:
                        simhash_results.append({
                            "doc_a": doc_a,
                            "doc_b": doc_b,
                            "similarity": round(sim, 6),
                            "hamming_distance": hamming,
                            "method": "simhash"
                        })
            
            simhash_results.sort(key=lambda x: x["similarity"], reverse=True)
            
            elapsed_simhash = time.time() - start_time
            
            print(f"\nSimHash results: {len(simhash_results)} pairs above threshold")
            print(f"Execution time: {elapsed_simhash:.3f}s")
            
            results.extend(simhash_results)
        
        # Output results
        print("\n" + "=" * 60)
        print("FINAL RESULTS")
        print("=" * 60)
        print(f"Total candidate pairs found: {len(results)}")
        
        if results:
            print(f"\nTop 10 most similar pairs:")
            sorted_results = sorted(results, key=lambda x: x["similarity"], reverse=True)
            for i, pair in enumerate(sorted_results[:10], 1):
                print(f"  {i}. {pair['doc_a']} <-> {pair['doc_b']}")
                print(f"     Similarity: {pair['similarity']:.4f} ({pair['method']})")
        
        self.write_csv(results, args.output)
    
    def compute_metrics(
        self,
        y_true: List[int],
        y_pred: List[int]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
            
        Returns:
            Dictionary with precision, recall, F1, accuracy
        """
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "accuracy": accuracy,
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn
        }
    
    def command_pairs(self, args) -> None:
        """
        Command 3: Evaluate on labeled pair dataset.
        
        Computes precision, recall, F1-score for different methods.
        
        Args:
            args: Parsed command-line arguments
        """
        print("=" * 60)
        print("COMMAND: Evaluate on Labeled Pairs")
        print("=" * 60)
        
        # Load dataset
        print(f"\nLoading dataset: {args.pairs}")
        try:
            df = pd.read_csv(args.pairs)
        except Exception as e:
            print(f"Error loading CSV: {e}", file=sys.stderr)
            sys.exit(1)
        
        # Validate columns
        required_cols = [args.text_col_a, args.text_col_b, args.label_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"Error: Missing columns: {missing_cols}", file=sys.stderr)
            print(f"Available columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(1)
        
        # Apply limit
        if args.limit and args.limit < len(df):
            df = df.head(args.limit)
            print(f"Limited to first {args.limit} pairs")
        
        print(f"Evaluating {len(df)} pairs")
        
        # Clean data
        df = df.dropna(subset=[args.text_col_a, args.text_col_b, args.label_col])
        print(f"After removing NaN: {len(df)} pairs")
        
        # Extract texts and labels
        texts_a = df[args.text_col_a].astype(str).tolist()
        texts_b = df[args.text_col_b].astype(str).tolist()
        labels = df[args.label_col].astype(int).tolist()
        
        # Initialize components
        shingler = WordShingles(k=args.shingle_size)
        minhash = MinHash(num_permutations=args.minhash_perms)
        simhash = SimHash(use_tfidf=True)
        
        # Fit SimHash on corpus
        all_texts = texts_a + texts_b
        simhash.fit(all_texts)
        
        metrics_results = []
        
        # Method 1: Exact Jaccard
        if args.method in ["jaccard", "all"]:
            print("\n" + "=" * 60)
            print("METHOD 1: Exact Jaccard Similarity")
            print("=" * 60)
            
            start_time = time.time()
            predictions = []
            
            for i, (text_a, text_b) in enumerate(zip(texts_a, texts_b)):
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(texts_a)} pairs...")
                
                shingles_a = shingler.generate_shingles(text_a)
                shingles_b = shingler.generate_shingles(text_b)
                sim = shingler.jaccard_similarity(shingles_a, shingles_b)
                
                predictions.append(1 if sim >= args.threshold else 0)
            
            elapsed = time.time() - start_time
            metrics = self.compute_metrics(labels, predictions)
            
            print(f"\nJaccard Results:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Time:      {elapsed:.3f}s")
            
            metrics_results.append({
                "method": "jaccard",
                "threshold": args.threshold,
                "precision": round(metrics['precision'], 6),
                "recall": round(metrics['recall'], 6),
                "f1_score": round(metrics['f1_score'], 6),
                "accuracy": round(metrics['accuracy'], 6),
                "true_positives": metrics['true_positives'],
                "false_positives": metrics['false_positives'],
                "true_negatives": metrics['true_negatives'],
                "false_negatives": metrics['false_negatives'],
                "execution_time_seconds": round(elapsed, 3)
            })
        
        # Method 2: MinHash
        if args.method in ["minhash", "all"]:
            print("\n" + "=" * 60)
            print("METHOD 2: MinHash Approximation")
            print("=" * 60)
            
            start_time = time.time()
            predictions = []
            
            for i, (text_a, text_b) in enumerate(zip(texts_a, texts_b)):
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(texts_a)} pairs...")
                
                shingles_a = shingler.generate_shingles(text_a)
                shingles_b = shingler.generate_shingles(text_b)
                
                sig_a = minhash.compute_signature(shingles_a)
                sig_b = minhash.compute_signature(shingles_b)
                
                sim = minhash.estimate_similarity(sig_a, sig_b)
                predictions.append(1 if sim >= args.threshold else 0)
            
            elapsed = time.time() - start_time
            metrics = self.compute_metrics(labels, predictions)
            
            print(f"\nMinHash Results:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Time:      {elapsed:.3f}s")
            
            metrics_results.append({
                "method": "minhash",
                "threshold": args.threshold,
                "signature_length": args.minhash_perms,
                "precision": round(metrics['precision'], 6),
                "recall": round(metrics['recall'], 6),
                "f1_score": round(metrics['f1_score'], 6),
                "accuracy": round(metrics['accuracy'], 6),
                "true_positives": metrics['true_positives'],
                "false_positives": metrics['false_positives'],
                "true_negatives": metrics['true_negatives'],
                "false_negatives": metrics['false_negatives'],
                "execution_time_seconds": round(elapsed, 3)
            })
        
        # Method 3: SimHash
        if args.method in ["simhash", "all"]:
            print("\n" + "=" * 60)
            print("METHOD 3: TF-IDF Weighted SimHash")
            print("=" * 60)
            
            start_time = time.time()
            predictions = []
            
            for i, (text_a, text_b) in enumerate(zip(texts_a, texts_b)):
                if (i + 1) % 500 == 0:
                    print(f"  Processed {i + 1}/{len(texts_a)} pairs...")
                
                fp_a = simhash.compute_fingerprint(text_a)
                fp_b = simhash.compute_fingerprint(text_b)
                
                sim = simhash.similarity(fp_a, fp_b)
                predictions.append(1 if sim >= args.threshold else 0)
            
            elapsed = time.time() - start_time
            metrics = self.compute_metrics(labels, predictions)
            
            print(f"\nSimHash Results:")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1-Score:  {metrics['f1_score']:.4f}")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Time:      {elapsed:.3f}s")
            
            metrics_results.append({
                "method": "simhash",
                "threshold": args.threshold,
                "hash_bits": SimHash.HASH_BITS,
                "precision": round(metrics['precision'], 6),
                "recall": round(metrics['recall'], 6),
                "f1_score": round(metrics['f1_score'], 6),
                "accuracy": round(metrics['accuracy'], 6),
                "true_positives": metrics['true_positives'],
                "false_positives": metrics['false_positives'],
                "true_negatives": metrics['true_negatives'],
                "false_negatives": metrics['false_negatives'],
                "execution_time_seconds": round(elapsed, 3)
            })
        
        # Output results
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        
        for result in metrics_results:
            print(f"\n{result['method'].upper()}:")
            print(f"  F1-Score:  {result['f1_score']:.4f}")
            print(f"  Precision: {result['precision']:.4f}")
            print(f"  Recall:    {result['recall']:.4f}")
            print(f"  Accuracy:  {result['accuracy']:.4f}")
            print(f"  Time:      {result['execution_time_seconds']:.3f}s")
        
        self.write_csv(metrics_results, args.output)
    
    def run(self) -> None:
        """
        Main entry point for CLI.
        """
        parser = self.setup_argparse()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            sys.exit(1)
        
        # Route to appropriate command handler
        if args.command == "compare":
            self.command_compare(args)
        elif args.command == "corpus":
            self.command_corpus(args)
        elif args.command == "pairs":
            self.command_pairs(args)
        else:
            print(f"Error: Unknown command: {args.command}", file=sys.stderr)
            sys.exit(1)


def main():
    """
    Main function for CLI execution.
    """
    cli = PlagiarismCLI()
    cli.run()


if __name__ == "__main__":
    main()
