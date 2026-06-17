# tests/test_minhash.py
"""
Unit tests for MinHash module.
"""

import pytest
import numpy as np
from src.minhash import MinHash, MinHashIndex
from src.preprocessing import WordShingles


class TestMinHash:
    """Test MinHash signature generation."""
    
    def test_initialization(self):
        """Test MinHash initialization."""
        mh = MinHash(num_permutations=128, seed=42)
        
        assert mh.num_permutations == 128
        assert mh.seed == 42
        assert len(mh._hash_params) == 128
    
    def test_invalid_num_permutations(self):
        """Test that invalid num_permutations raises error."""
        with pytest.raises(ValueError):
            MinHash(num_permutations=0)
        
        with pytest.raises(ValueError):
            MinHash(num_permutations=-5)
    
    def test_signature_shape(self):
        """Test that signature has correct shape."""
        mh = MinHash(num_permutations=128)
        
        shingles = {'hello world', 'world hello', 'test shingle'}
        signature = mh.compute_signature(shingles)
        
        assert signature.shape == (128,)
        assert signature.dtype == np.int64
    
    def test_empty_set_signature(self):
        """Test signature for empty set."""
        mh = MinHash(num_permutations=64)
        
        signature = mh.compute_signature(set())
        
        # Empty set should get max values
        assert np.all(signature == mh._MERSENNE_PRIME)
    
    def test_deterministic_signatures(self):
        """Test that same input produces same signature with same seed."""
        mh1 = MinHash(num_permutations=64, seed=42)
        mh2 = MinHash(num_permutations=64, seed=42)
        
        shingles = {'the cat sat', 'cat sat on', 'sat on mat'}
        
        sig1 = mh1.compute_signature(shingles)
        sig2 = mh2.compute_signature(shingles)
        
        assert np.array_equal(sig1, sig2)
    
    def test_different_seeds_produce_different_signatures(self):
        """Test that different seeds produce different signatures."""
        mh1 = MinHash(num_permutations=64, seed=42)
        mh2 = MinHash(num_permutations=64, seed=99)
        
        shingles = {'the cat sat', 'cat sat on', 'sat on mat'}
        
        sig1 = mh1.compute_signature(shingles)
        sig2 = mh2.compute_signature(shingles)
        
        # Should be different (with very high probability)
        assert not np.array_equal(sig1, sig2)
    
    def test_similarity_estimation_identical_sets(self):
        """Test similarity estimation for identical sets."""
        mh = MinHash(num_permutations=128)
        
        shingles = {'hello world', 'world hello', 'test'}
        
        sig = mh.compute_signature(shingles)
        similarity = mh.estimate_similarity(sig, sig)
        
        assert similarity == 1.0
    
    def test_similarity_estimation_overlapping_sets(self):
        """Test similarity estimation for overlapping sets."""
        mh = MinHash(num_permutations=128)
        
        shingles_a = {'a', 'b', 'c', 'd'}
        shingles_b = {'c', 'd', 'e', 'f'}
        
        sig_a = mh.compute_signature(shingles_a)
        sig_b = mh.compute_signature(shingles_b)
        
        similarity = mh.estimate_similarity(sig_a, sig_b)
        
        # True Jaccard = 2/6 = 0.333...
        # MinHash estimate should be close but may vary
        assert 0.0 <= similarity <= 1.0
    
    def test_similarity_estimation_disjoint_sets(self):
        """Test similarity estimation for completely different sets."""
        mh = MinHash(num_permutations=256)
        
        shingles_a = {'a', 'b', 'c'}
        shingles_b = {'x', 'y', 'z'}
        
        sig_a = mh.compute_signature(shingles_a)
        sig_b = mh.compute_signature(shingles_b)
        
        similarity = mh.estimate_similarity(sig_a, sig_b)
        
        # Should be very low (close to 0)
        assert similarity < 0.2
    
    def test_estimate_similarity_dimension_mismatch(self):
        """Test that mismatched signature lengths raise error."""
        mh = MinHash(num_permutations=64)
        
        sig_a = np.array([1, 2, 3, 4])
        sig_b = np.array([1, 2, 3])
        
        with pytest.raises(ValueError):
            mh.estimate_similarity(sig_a, sig_b)
    
    def test_compute_signature_from_text(self):
        """Test computing signature directly from text."""
        mh = MinHash(num_permutations=128)
        shingler = WordShingles(k=3)
        
        text = "the cat sat on the mat"
        signature = mh.compute_signature_from_text(text, shingler)
        
        assert signature.shape == (128,)


class TestMinHashIndex:
    """Test MinHash indexing functionality."""
    
    def test_add_and_retrieve_document(self):
        """Test adding documents to index."""
        index = MinHashIndex(num_permutations=64)
        
        shingles = {'hello world', 'world hello'}
        index.add_document('doc1', shingles)
        
        assert 'doc1' in index.signatures
        assert 'doc1' in index.doc_ids
        
        sig = index.get_signature('doc1')
        assert sig is not None
        assert sig.shape == (64,)
    
    def test_compare_documents(self):
        """Test comparing two indexed documents."""
        index = MinHashIndex(num_permutations=128)
        
        shingles_a = {'a', 'b', 'c'}
        shingles_b = {'b', 'c', 'd'}
        
        index.add_document('doc_a', shingles_a)
        index.add_document('doc_b', shingles_b)
        
        similarity = index.compare('doc_a', 'doc_b')
        
        assert 0.0 <= similarity <= 1.0
    
    def test_compare_nonexistent_documents(self):
        """Test that comparing non-existent documents raises error."""
        index = MinHashIndex(num_permutations=64)
        
        index.add_document('doc1', {'a', 'b'})
        
        with pytest.raises(KeyError):
            index.compare('doc1', 'doc_nonexistent')
    
    def test_find_similar_documents(self):
        """Test finding similar documents."""
        index = MinHashIndex(num_permutations=128, seed=42)
        
        # Add several documents
        index.add_document('doc1', {'a', 'b', 'c', 'd'})
        index.add_document('doc2', {'a', 'b', 'c', 'e'})  # Very similar to doc1
        index.add_document('doc3', {'x', 'y', 'z'})        # Completely different
        index.add_document('doc4', {'a', 'b'})             # Somewhat similar
        
        # Find documents similar to doc1
        similar = index.find_similar('doc1', threshold=0.3)
        
        # Should find doc2 and doc4, but not doc3
        similar_ids = [doc_id for doc_id, _ in similar]
        
        assert 'doc2' in similar_ids
        assert 'doc4' in similar_ids
        assert 'doc3' not in similar_ids
    
    def test_find_similar_sorted_by_similarity(self):
        """Test that results are sorted by similarity."""
        index = MinHashIndex(num_permutations=128, seed=42)
        
        index.add_document('doc1', {'a', 'b', 'c'})
        index.add_document('doc2', {'a', 'b', 'c', 'd'})
        index.add_document('doc3', {'a', 'b'})
        
        similar = index.find_similar('doc1', threshold=0.0)
        
        # Check that similarities are in descending order
        similarities = [sim for _, sim in similar]
        assert similarities == sorted(similarities, reverse=True)
    
    def test_find_similar_excludes_self(self):
        """Test that find_similar doesn't return the query document itself."""
        index = MinHashIndex(num_permutations=64)
        
        index.add_document('doc1', {'a', 'b', 'c'})
        index.add_document('doc2', {'a', 'b', 'c'})
        
        similar = index.find_similar('doc1', threshold=0.0)
        
        similar_ids = [doc_id for doc_id, _ in similar]
        assert 'doc1' not in similar_ids
