# src/simhash.py
"""
SimHash implementation for document similarity detection.

Based on DM_P3_Guide.pdf (Page 3):
- Documents are transformed into tokens/n-grams
- Each token is weighted by TF-IDF
- Using 64-bit hash, weighted contributions accumulate into 64-dimensional vector
- Final bit in each dimension determined by sign of accumulated value
- Similar documents have small Hamming distance
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import re


class SimHash:
    """
    SimHash fingerprint generator for documents.
    
    SimHash creates a compact binary fingerprint (64 bits) that preserves
    similarity: similar documents have similar fingerprints (small Hamming distance).
    """
    
    HASH_BITS = 64  # 64-bit hash as per guide
    
    def __init__(self, use_tfidf: bool = True):
        """
        Initialize SimHash generator.
        
        Args:
            use_tfidf: Whether to use TF-IDF weighting (default: True as per guide)
        """
        self.use_tfidf = use_tfidf
        
        # For TF-IDF calculation
        self.document_count = 0
        self.token_doc_count: Dict[str, int] = {}  # How many docs contain each token
    
    def _hash_token(self, token: str) -> int:
        """
        Hash a token to a 64-bit integer.
        
        Args:
            token: Input token string
            
        Returns:
            64-bit hash value
        """
        # Use Python's built-in hash and mask to 64 bits
        h = hash(token)
        return h & ((1 << self.HASH_BITS) - 1)
    
    def _get_bit(self, hash_value: int, bit_position: int) -> int:
        """
        Extract a specific bit from hash value.
        
        Args:
            hash_value: 64-bit hash
            bit_position: Position (0-63)
            
        Returns:
            1 if bit is set, -1 if not (for accumulation)
        """
        if (hash_value >> bit_position) & 1:
            return 1
        else:
            return -1
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Clean and lowercase
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Split into tokens
        tokens = text.split()
        
        return tokens
    
    def compute_tf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Compute term frequency for tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary mapping token to TF score
        """
        if not tokens:
            return {}
        
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        # TF = count / total
        tf = {token: count / total_tokens for token, count in token_counts.items()}
        
        return tf
    
    def compute_idf(self, token: str, total_docs: int) -> float:
        """
        Compute inverse document frequency for a token.
        
        Args:
            token: Token string
            total_docs: Total number of documents
            
        Returns:
            IDF score
        """
        if total_docs == 0:
            return 0.0
        
        # Number of documents containing this token
        doc_count = self.token_doc_count.get(token, 0)
        
        if doc_count == 0:
            return 0.0
        
        # IDF = log(N / df)
        idf = np.log(total_docs / doc_count)
        
        return idf
    
    def compute_tfidf(self, tokens: List[str]) -> Dict[str, float]:
        """
        Compute TF-IDF weights for tokens.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Dictionary mapping token to TF-IDF weight
        """
        tf = self.compute_tf(tokens)
        
        if not self.use_tfidf or self.document_count == 0:
            # Just use TF
            return tf
        
        # Compute TF-IDF
        tfidf = {}
        for token, tf_score in tf.items():
            idf = self.compute_idf(token, self.document_count)
            tfidf[token] = tf_score * idf
        
        return tfidf
    
    def compute_fingerprint(self, text: str, tokens: Optional[List[str]] = None) -> int:
        """
        Compute SimHash fingerprint for text.
        
        According to the guide (Page 3):
        1. Tokenize document
        2. Compute TF-IDF for each token
        3. For each token, hash to 64 bits
        4. Accumulate weighted contributions into 64-dimensional vector
        5. Final bit determined by sign of accumulated value
        
        Args:
            text: Input text
            tokens: Pre-computed tokens (optional, will tokenize if None)
            
        Returns:
            64-bit SimHash fingerprint
        """
        if tokens is None:
            tokens = self.tokenize(text)
        
        if not tokens:
            return 0
        
        # Compute weights (TF-IDF or just TF)
        weights = self.compute_tfidf(tokens)
        
        # Initialize accumulator vector (64 dimensions)
        accumulator = np.zeros(self.HASH_BITS, dtype=np.float64)
        
        # For each unique token
        for token, weight in weights.items():
            # Hash the token
            token_hash = self._hash_token(token)
            
            # For each bit position
            for bit_pos in range(self.HASH_BITS):
                # Get bit value (+1 or -1)
                bit_value = self._get_bit(token_hash, bit_pos)
                
                # Accumulate weighted contribution
                accumulator[bit_pos] += weight * bit_value
        
        # Generate final fingerprint based on sign
        fingerprint = 0
        for bit_pos in range(self.HASH_BITS):
            if accumulator[bit_pos] >= 0:
                # Set bit to 1
                fingerprint |= (1 << bit_pos)
        
        return fingerprint
    
    def hamming_distance(self, fp1: int, fp2: int) -> int:
        """
        Compute Hamming distance between two fingerprints.
        
        Hamming distance = number of differing bits
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Hamming distance (0 to 64)
        """
        # XOR gives 1 where bits differ
        xor = fp1 ^ fp2
        
        # Count set bits
        distance = bin(xor).count('1')
        
        return distance
    
    def similarity(self, fp1: int, fp2: int) -> float:
        """
        Compute similarity score from Hamming distance.
        
        Similarity = 1 - (hamming_distance / hash_bits)
        
        Args:
            fp1: First fingerprint
            fp2: Second fingerprint
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        distance = self.hamming_distance(fp1, fp2)
        return 1.0 - (distance / self.HASH_BITS)
    
    def fit(self, documents: List[str]) -> None:
        """
        Fit the SimHash model on a corpus for TF-IDF calculation.
        
        Args:
            documents: List of document texts
        """
        self.document_count = len(documents)
        self.token_doc_count.clear()
        
        # Count how many documents contain each token
        for doc in documents:
            tokens = self.tokenize(doc)
            unique_tokens = set(tokens)
            
            for token in unique_tokens:
                self.token_doc_count[token] = self.token_doc_count.get(token, 0) + 1


class SimHashIndex:
    """
    Index for managing multiple SimHash fingerprints.
    """
    
    def __init__(self, use_tfidf: bool = True):
        """
        Initialize SimHash index.
        
        Args:
            use_tfidf: Whether to use TF-IDF weighting
        """
        self.simhash = SimHash(use_tfidf=use_tfidf)
        self.fingerprints: Dict[str, int] = {}
        self.doc_ids: List[str] = []
    
    def fit(self, documents: Dict[str, str]) -> None:
        """
        Fit the index on a corpus for TF-IDF.
        
        Args:
            documents: Dictionary mapping doc_id to text
        """
        self.simhash.fit(list(documents.values()))
    
    def add_document(self, doc_id: str, text: str) -> None:
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            text: Document text
        """
        fingerprint = self.simhash.compute_fingerprint(text)
        self.fingerprints[doc_id] = fingerprint
        
        if doc_id not in self.doc_ids:
            self.doc_ids.append(doc_id)
    
    def get_fingerprint(self, doc_id: str) -> Optional[int]:
        """
        Get fingerprint for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Fingerprint or None if not found
        """
        return self.fingerprints.get(doc_id)
    
    def compare(self, doc_id_a: str, doc_id_b: str) -> float:
        """
        Compare two indexed documents.
        
        Args:
            doc_id_a: First document ID
            doc_id_b: Second document ID
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        fp_a = self.fingerprints.get(doc_id_a)
        fp_b = self.fingerprints.get(doc_id_b)
        
        if fp_a is None or fp_b is None:
            raise KeyError("One or both documents not found in index")
        
        return self.simhash.similarity(fp_a, fp_b)
    
    def find_similar(
        self,
        doc_id: str,
        threshold: float = 0.8,
        max_hamming_distance: Optional[int] = None
    ) -> List[Tuple[str, float, int]]:
        """
        Find similar documents to the given document.
        
        Args:
            doc_id: Target document ID
            threshold: Minimum similarity threshold
            max_hamming_distance: Maximum Hamming distance (optional)
            
        Returns:
            List of (doc_id, similarity, hamming_distance) tuples, sorted by similarity
        """
        target_fp = self.fingerprints.get(doc_id)
        if target_fp is None:
            raise KeyError(f"Document {doc_id} not found in index")
        
        results = []
        
        for other_id in self.doc_ids:
            if other_id == doc_id:
                continue
            
            other_fp = self.fingerprints[other_id]
            hamming = self.simhash.hamming_distance(target_fp, other_fp)
            similarity = self.simhash.similarity(target_fp, other_fp)
            
            # Apply filters
            if max_hamming_distance is not None and hamming > max_hamming_distance:
                continue
            
            if similarity >= threshold:
                results.append((other_id, similarity, hamming))
        
        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
