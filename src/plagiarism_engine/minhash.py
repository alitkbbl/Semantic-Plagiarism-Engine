# src/minhash.py
"""
MinHash signature generation for approximate Jaccard similarity.

Implementation from scratch as required by the guide (Page 3):
- No use of datasketch or similar libraries for the core implementation
- Signature length: 128 or 256 (configurable)
- Similarity estimated by fraction of equal positions in signatures
"""

import numpy as np
from typing import Set, List, Tuple


class MinHash:
    """
    MinHash signature generator.
    
    MinHash provides an efficient way to estimate Jaccard similarity
    by creating compact signatures from sets.
    """
    
    # Large prime number for hash function generation
    _MERSENNE_PRIME = (1 << 61) - 1  # 2^61 - 1
    _MAX_HASH = (1 << 32) - 1  # Maximum 32-bit hash value
    
    def __init__(self, num_permutations: int = 128, seed: int = 42):
        """
        Initialize MinHash signature generator.
        
        Args:
            num_permutations: Number of hash functions (signature length)
                             Recommended: 128 or 256
            seed: Random seed for reproducibility
        """
        if num_permutations < 1:
            raise ValueError(f"num_permutations must be >= 1, got {num_permutations}")
        
        self.num_permutations = num_permutations
        self.seed = seed
        
        # Generate hash function parameters
        self._hash_params = self._generate_hash_functions()
    
    def _generate_hash_functions(self) -> List[Tuple[int, int]]:
        """
        Generate parameters for hash functions.
        
        We use the form: h(x) = (a * x + b) mod prime
        where a and b are random coefficients.
        
        Returns:
            List of (a, b) tuples for each hash function
        """
        rng = np.random.RandomState(self.seed)
        
        hash_params = []
        for _ in range(self.num_permutations):
            a = rng.randint(1, self._MERSENNE_PRIME, dtype=np.int64)
            b = rng.randint(0, self._MERSENNE_PRIME, dtype=np.int64)
            hash_params.append((a, b))
        
        return hash_params
    
    def _hash_string(self, s: str) -> int:
        """
        Hash a string to a 32-bit integer.
        
        Args:
            s: Input string (shingle)
            
        Returns:
            32-bit hash value
        """
        # Use Python's built-in hash, but constrain to 32 bits
        h = hash(s)
        return h & self._MAX_HASH
    
    def _min_hash_value(self, element_hash: int, a: int, b: int) -> int:
        """
        Apply a single hash function to an element.
        
        Args:
            element_hash: Hash of the element
            a: First hash parameter
            b: Second hash parameter
            
        Returns:
            Transformed hash value
        """
        return (a * element_hash + b) % self._MERSENNE_PRIME
    
    def compute_signature(self, shingle_set: Set[str]) -> np.ndarray:
        """
        Compute MinHash signature for a set of shingles.
        
        The signature is a vector where each position holds the minimum
        hash value across all shingles for that particular hash function.
        
        Args:
            shingle_set: Set of shingles (strings)
            
        Returns:
            MinHash signature as numpy array of shape (num_permutations,)
        """
        if not shingle_set:
            # Empty set gets signature of maximum values
            return np.full(self.num_permutations, self._MERSENNE_PRIME, dtype=np.int64)
        
        # Initialize signature with maximum values
        signature = np.full(self.num_permutations, self._MERSENNE_PRIME, dtype=np.int64)
        
        # For each shingle
        for shingle in shingle_set:
            # Hash the shingle once
            element_hash = self._hash_string(shingle)
            
            # Apply each hash function and update signature
            for i, (a, b) in enumerate(self._hash_params):
                hash_value = self._min_hash_value(element_hash, a, b)
                
                # Keep the minimum
                if hash_value < signature[i]:
                    signature[i] = hash_value
        
        return signature
    
    def compute_signature_from_text(self, text: str, shingler) -> np.ndarray:
        """
        Convenience method: compute signature directly from text.
        
        Args:
            text: Input text
            shingler: WordShingles instance for generating shingles
            
        Returns:
            MinHash signature
        """
        shingles = shingler.generate_shingles(text)
        return self.compute_signature(shingles)
    
    def estimate_similarity(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """
        Estimate Jaccard similarity from two MinHash signatures.
        
        According to the guide (Page 3):
        Similarity = fraction of equal positions in the two signatures
        
        Args:
            sig_a: First MinHash signature
            sig_b: Second MinHash signature
            
        Returns:
            Estimated Jaccard similarity (0.0 to 1.0)
        """
        if len(sig_a) != len(sig_b):
            raise ValueError(
                f"Signatures must have same length: {len(sig_a)} vs {len(sig_b)}"
            )
        
        if len(sig_a) == 0:
            return 0.0
        
        # Count equal positions
        equal_count = np.sum(sig_a == sig_b)
        
        # Return fraction
        return equal_count / len(sig_a)


class MinHashIndex:
    """
    Index multiple documents with MinHash signatures for batch comparison.
    """
    
    def __init__(self, num_permutations: int = 128, seed: int = 42):
        """
        Initialize MinHash index.
        
        Args:
            num_permutations: Number of hash functions (signature length)
            seed: Random seed for reproducibility
        """
        self.minhash = MinHash(num_permutations=num_permutations, seed=seed)
        self.signatures = {}  # doc_id -> signature
        self.doc_ids = []     # Ordered list of document IDs
    
    def add_document(self, doc_id: str, shingle_set: Set[str]) -> None:
        """
        Add a document to the index.
        
        Args:
            doc_id: Unique document identifier
            shingle_set: Set of shingles for the document
        """
        signature = self.minhash.compute_signature(shingle_set)
        self.signatures[doc_id] = signature
        if doc_id not in self.doc_ids:
            self.doc_ids.append(doc_id)
    
    def get_signature(self, doc_id: str) -> np.ndarray:
        """
        Retrieve signature for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            MinHash signature
        """
        return self.signatures.get(doc_id)
    
    def compare(self, doc_id_a: str, doc_id_b: str) -> float:
        """
        Compare two indexed documents.
        
        Args:
            doc_id_a: First document ID
            doc_id_b: Second document ID
            
        Returns:
            Estimated Jaccard similarity
        """
        sig_a = self.signatures.get(doc_id_a)
        sig_b = self.signatures.get(doc_id_b)
        
        if sig_a is None or sig_b is None:
            raise KeyError(f"One or both documents not found in index")
        
        return self.minhash.estimate_similarity(sig_a, sig_b)
    
    def find_similar(self, doc_id: str, threshold: float = 0.5) -> List[Tuple[str, float]]:
        """
        Find all documents similar to the given document above a threshold.
        
        Args:
            doc_id: Target document ID
            threshold: Minimum similarity threshold
            
        Returns:
            List of (doc_id, similarity) tuples, sorted by similarity (descending)
        """
        target_sig = self.signatures.get(doc_id)
        if target_sig is None:
            raise KeyError(f"Document {doc_id} not found in index")
        
        results = []
        for other_id in self.doc_ids:
            if other_id == doc_id:
                continue  # Skip self-comparison
            
            similarity = self.minhash.estimate_similarity(
                target_sig,
                self.signatures[other_id]
            )
            
            if similarity >= threshold:
                results.append((other_id, similarity))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
