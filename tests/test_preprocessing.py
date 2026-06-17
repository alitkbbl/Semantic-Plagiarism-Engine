# tests/test_preprocessing.py
"""
Unit tests for preprocessing module.
"""

import pytest
from src.preprocessing import TextPreprocessor, WordShingles, create_shingles, jaccard_similarity


class TestTextPreprocessor:
    """Test text preprocessing functionality."""
    
    def test_basic_cleaning(self):
        """Test basic text cleaning."""
        preprocessor = TextPreprocessor()
        
        text = "  Hello,  World!  How are   you?  "
        cleaned = preprocessor.clean(text)
        
        assert cleaned == "hello world how are you"
    
    def test_empty_text(self):
        """Test handling of empty text."""
        preprocessor = TextPreprocessor()
        
        assert preprocessor.clean("") == ""
        assert preprocessor.clean("   ") == ""
        assert preprocessor.clean(None) == ""
    
    def test_punctuation_removal(self):
        """Test punctuation removal."""
        preprocessor = TextPreprocessor(remove_punctuation=True)
        
        text = "Hello! How's it going? Fine, thanks."
        cleaned = preprocessor.clean(text)
        
        assert "!" not in cleaned
        assert "?" not in cleaned
        assert "," not in cleaned
        assert "'" not in cleaned


class TestWordShingles:
    """Test word shingling functionality."""
    
    def test_basic_shingling(self):
        """Test basic k-shingle generation."""
        shingler = WordShingles(k=3)
        
        text = "the cat sat on the mat"
        shingles = shingler.generate_shingles(text)
        
        expected = {
            'the cat sat',
            'cat sat on',
            'sat on the',
            'on the mat'
        }
        
        assert shingles == expected
    
    def test_short_text(self):
        """Test handling of text shorter than k."""
        shingler = WordShingles(k=3)
        
        text = "hello world"
        shingles = shingler.generate_shingles(text)
        
        # Should return entire text as one shingle
        assert len(shingles) == 1
        assert "hello world" in shingles
    
    def test_empty_text(self):
        """Test handling of empty text."""
        shingler = WordShingles(k=3)
        
        shingles = shingler.generate_shingles("")
        assert len(shingles) == 0
    
    def test_different_k_values(self):
        """Test shingling with different k values."""
        text = "the quick brown fox jumps"
        
        shingler_2 = WordShingles(k=2)
        shingles_2 = shingler_2.generate_shingles(text)
        assert len(shingles_2) == 4
        
        shingler_4 = WordShingles(k=4)
        shingles_4 = shingler_4.generate_shingles(text)
        assert len(shingles_4) == 2
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity calculation."""
        shingler = WordShingles(k=2)
        
        set_a = {'a b', 'b c', 'c d'}
        set_b = {'b c', 'c d', 'd e'}
        
        similarity = shingler.jaccard_similarity(set_a, set_b)
        
        # Intersection: {b c, c d} = 2
        # Union: {a b, b c, c d, d e} = 4
        # J(A,B) = 2/4 = 0.5
        assert similarity == 0.5
    
    def test_jaccard_identical_sets(self):
        """Test Jaccard similarity of identical sets."""
        shingler = WordShingles(k=2)
        
        set_a = {'a b', 'b c'}
        
        similarity = shingler.jaccard_similarity(set_a, set_a)
        assert similarity == 1.0
    
    def test_jaccard_disjoint_sets(self):
        """Test Jaccard similarity of completely different sets."""
        shingler = WordShingles(k=2)
        
        set_a = {'a b', 'b c'}
        set_b = {'x y', 'y z'}
        
        similarity = shingler.jaccard_similarity(set_a, set_b)
        assert similarity == 0.0
    
    def test_jaccard_empty_sets(self):
        """Test Jaccard similarity with empty sets."""
        shingler = WordShingles(k=2)
        
        # Both empty = identical
        similarity = shingler.jaccard_similarity(set(), set())
        assert similarity == 1.0


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    def test_create_shingles(self):
        """Test create_shingles convenience function."""
        shingles = create_shingles("the cat sat", k=2)
        
        assert 'the cat' in shingles
        assert 'cat sat' in shingles
    
    def test_jaccard_similarity_function(self):
        """Test jaccard_similarity convenience function."""
        text_a = "the cat sat on the mat"
        text_b = "the cat sat on a mat"
        
        similarity = jaccard_similarity(text_a, text_b, k=3)
        
        # Should be high but not 1.0 (difference: "the mat" vs "a mat")
        assert 0.5 < similarity < 1.0
