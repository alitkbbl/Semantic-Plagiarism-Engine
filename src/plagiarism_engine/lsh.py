# src/lsh.py
"""
Locality-Sensitive Hashing (LSH) for efficient similarity search.

Based on DM_P3_Guide.pdf (Page 3):
- MinHash signatures are split into bands
- Each band is hashed into buckets
- Documents sharing at least one bucket become candidate pairs
- This reduces the number of comparisons vs. all-to-all approach
"""

import numpy as np
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict


class LSH:
    """
    Locality-Sensitive Hashing for MinHash signatures.
    
    LSH divides MinHash signatures into bands and hashes each band.
    Documents that hash to the same bucket in any band become candidates
    for similarity comparison.
    """
    
    def __init__(self, num_bands: int, rows_per_band: int):
        """
        Initialize LSH index.
        
        Args:
            num_bands: Number of bands to split signature into
            rows_per_band: Number of hash values per band
            
        Note:
            signature_length = num_bands × rows_per_band
            For MinHash with 128 permutations:
                - 32 bands × 4 rows = 128
                - 16 bands × 8 rows = 128
                - 8 bands × 16 rows = 128
        """
        if num_bands < 1:
            raise ValueError(f"num_bands must be >= 1, got {num_bands}")
        if rows_per_band < 1:
            raise ValueError(f"rows_per_band must be >= 1, got {rows_per_band}")
        
        self.num_bands = num_bands
        self.rows_per_band = rows_per_band
        self.signature_length = num_bands * rows_per_band
        
        # Bucket storage: band_idx -> bucket_hash -> set of doc_ids
        self.buckets: Dict[int, Dict[int, Set[str]]] = defaultdict(lambda: defaultdict(set))
        
        # Document signatures
        self.signatures: Dict[str, np.ndarray] = {}
    
    def _hash_band(self, band: np.ndarray) -> int:
        """
        Hash a band to a bucket.
        
        Args:
            band: Array of hash values for one band
            
        Returns:
            Bucket hash value
        """
        # Convert band to tuple for hashing
        return hash(tuple(band))
    
    def add_document(self, doc_id: str, signature: np.ndarray) -> None:
        """
        Add a document's MinHash signature to the LSH index.
        
        Args:
            doc_id: Unique document identifier
            signature: MinHash signature (must match signature_length)
            
        Raises:
            ValueError: If signature length doesn't match
        """
        if len(signature) != self.signature_length:
            raise ValueError(
                f"Signature length {len(signature)} doesn't match "
                f"expected {self.signature_length} "
                f"(num_bands={self.num_bands} × rows_per_band={self.rows_per_band})"
            )
        
        # Store signature
        self.signatures[doc_id] = signature
        
        # Split signature into bands and hash each band
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            
            # Hash the band to a bucket
            bucket_hash = self._hash_band(band)
            
            # Add document to this bucket
            self.buckets[band_idx][bucket_hash].add(doc_id)
    
    def get_candidate_pairs(self) -> Set[Tuple[str, str]]:
        """
        Get all candidate pairs that share at least one bucket.
        
        According to the guide (Page 3):
        If two documents share at least one bucket, they become candidate pairs.
        
        Returns:
            Set of (doc_id1, doc_id2) tuples where doc_id1 < doc_id2
        """
        candidates = set()
        
        # For each band
        for band_idx in range(self.num_bands):
            # For each bucket in this band
            for bucket_hash, doc_ids in self.buckets[band_idx].items():
                # If multiple documents in same bucket, they're candidates
                if len(doc_ids) > 1:
                    doc_list = sorted(doc_ids)
                    
                    # Generate all pairs from this bucket
                    for i in range(len(doc_list)):
                        for j in range(i + 1, len(doc_list)):
                            candidates.add((doc_list[i], doc_list[j]))
        
        return candidates
    
    def get_candidates_for_document(self, doc_id: str) -> Set[str]:
        """
        Get all candidate documents similar to the given document.
        
        Args:
            doc_id: Target document ID
            
        Returns:
            Set of candidate document IDs (excluding the query document itself)
        """
        if doc_id not in self.signatures:
            raise KeyError(f"Document {doc_id} not found in LSH index")
        
        candidates = set()
        signature = self.signatures[doc_id]
        
        # Check each band
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            
            # Hash the band
            bucket_hash = self._hash_band(band)
            
            # Get all documents in this bucket
            bucket_docs = self.buckets[band_idx].get(bucket_hash, set())
            candidates.update(bucket_docs)
        
        # Remove the query document itself
        candidates.discard(doc_id)
        
        return candidates
    
    def query(
        self,
        doc_id: str,
        minhash_calculator,
        threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find similar documents using LSH + MinHash verification.
        
        Args:
            doc_id: Query document ID
            minhash_calculator: MinHash instance to compute actual similarity
            threshold: Minimum similarity threshold for results
            
        Returns:
            List of (doc_id, similarity) tuples, sorted by similarity descending
        """
        # Get candidates using LSH
        candidates = self.get_candidates_for_document(doc_id)
        
        # Verify candidates using actual MinHash similarity
        results = []
        query_sig = self.signatures[doc_id]
        
        for candidate_id in candidates:
            candidate_sig = self.signatures[candidate_id]
            similarity = minhash_calculator.estimate_similarity(query_sig, candidate_sig)
            
            if similarity >= threshold:
                results.append((candidate_id, similarity))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def get_statistics(self) -> Dict:
        """
        Get LSH index statistics for analysis.
        
        Returns:
            Dictionary with statistics about the index
        """
        total_buckets = 0
        total_occupied_buckets = 0
        max_bucket_size = 0
        bucket_sizes = []
        
        for band_idx in range(self.num_bands):
            band_buckets = self.buckets[band_idx]
            total_buckets += len(band_buckets)
            
            for bucket_docs in band_buckets.values():
                size = len(bucket_docs)
                if size > 0:
                    total_occupied_buckets += 1
                    max_bucket_size = max(max_bucket_size, size)
                    bucket_sizes.append(size)
        
        num_docs = len(self.signatures)
        candidate_pairs = len(self.get_candidate_pairs())
        all_pairs = (num_docs * (num_docs - 1)) // 2
        
        return {
            'num_documents': num_docs,
            'num_bands': self.num_bands,
            'rows_per_band': self.rows_per_band,
            'signature_length': self.signature_length,
            'total_buckets': total_buckets,
            'occupied_buckets': total_occupied_buckets,
            'max_bucket_size': max_bucket_size,
            'avg_bucket_size': np.mean(bucket_sizes) if bucket_sizes else 0,
            'candidate_pairs': candidate_pairs,
            'all_possible_pairs': all_pairs,
            'reduction_ratio': 1 - (candidate_pairs / all_pairs) if all_pairs > 0 else 0
        }


def choose_lsh_parameters(
    signature_length: int,
    target_threshold: float = 0.5
) -> Tuple[int, int]:
    """
    Choose LSH parameters (bands and rows) based on signature length and threshold.
    
    The probability that two documents with Jaccard similarity s become candidates:
    P(candidate) = 1 - (1 - s^r)^b
    
    where:
        b = number of bands
        r = rows per band
        s = Jaccard similarity
    
    Args:
        signature_length: Length of MinHash signature
        target_threshold: Target similarity threshold (0.0 to 1.0)
        
    Returns:
        Tuple of (num_bands, rows_per_band)
    """
    # Find divisors of signature_length
    divisors = []
    for b in range(1, signature_length + 1):
        if signature_length % b == 0:
            r = signature_length // b
            divisors.append((b, r))
    
    # Choose parameters close to target threshold
    # The S-curve crosses 0.5 when s ≈ (1/b)^(1/r)
    best_params = (1, signature_length)
    best_diff = float('inf')
    
    for b, r in divisors:
        # Approximate threshold where P ≈ 0.5
        threshold_estimate = (1.0 / b) ** (1.0 / r)
        diff = abs(threshold_estimate - target_threshold)
        
        if diff < best_diff:
            best_diff = diff
            best_params = (b, r)
    
    return best_params
