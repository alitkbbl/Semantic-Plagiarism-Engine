# src/preprocessing.py
"""
Preprocessing module for document preparation and shingling.

This module handles:
- Text cleaning and normalization
- Word shingle generation (k-shingles)
- Shingle set creation for Jaccard similarity
"""

import re
from typing import Set, List, Optional


class TextPreprocessor:
    """
    Handles text cleaning and normalization.
    """
    
    def __init__(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_extra_whitespace: bool = True
    ):
        """
        Initialize the text preprocessor.
        
        Args:
            lowercase: Convert text to lowercase
            remove_punctuation: Remove punctuation marks
            remove_extra_whitespace: Normalize whitespace
        """
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.remove_extra_whitespace = remove_extra_whitespace
    
    def clean(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove punctuation
        if self.remove_punctuation:
            # Keep only alphanumeric and whitespace
            text = re.sub(r'[^\w\s]', ' ', text)
        
        # Normalize whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text


class WordShingles:
    """
    Generate word k-shingles from documents.
    
    According to the guide (Page 2):
    - Suggested shingle size: 3 to 5 words
    - Must handle very short, empty, or unusual-character texts
    """
    
    def __init__(self, k: int = 3, preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize the word shingling generator.
        
        Args:
            k: Size of word shingles (default: 3, recommended: 3-5)
            preprocessor: TextPreprocessor instance (if None, uses default)
        """
        if k < 1:
            raise ValueError(f"Shingle size k must be >= 1, got {k}")
        
        self.k = k
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of word tokens
        """
        # Clean text first
        text = self.preprocessor.clean(text)
        
        # Split into words
        tokens = text.split()
        
        return tokens
    
    def generate_shingles(self, text: str) -> Set[str]:
        """
        Generate k-word shingles from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of k-word shingles
            
        Example:
            >>> shingler = WordShingles(k=3)
            >>> shingler.generate_shingles("the cat sat on the mat")
            {'the cat sat', 'cat sat on', 'sat on the', 'on the mat'}
        """
        tokens = self.tokenize(text)
        
        # Handle edge cases
        if len(tokens) == 0:
            return set()
        
        if len(tokens) < self.k:
            # For very short documents, return the entire document as one shingle
            return {' '.join(tokens)}
        
        # Generate k-shingles
        shingles = set()
        for i in range(len(tokens) - self.k + 1):
            shingle = ' '.join(tokens[i:i + self.k])
            shingles.add(shingle)
        
        return shingles
    
    def jaccard_similarity(self, set_a: Set[str], set_b: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two shingle sets.
        
        According to the guide (Page 2):
        J(A, B) = |A ∩ B| / |A ∪ B|
        
        Args:
            set_a: First shingle set
            set_b: Second shingle set
            
        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        if len(set_a) == 0 and len(set_b) == 0:
            return 1.0  # Both empty = identical
        
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        
        if union == 0:
            return 0.0
        
        return intersection / union


# Convenience functions for quick usage
def create_shingles(text: str, k: int = 3) -> Set[str]:
    """
    Convenience function to generate shingles from text.
    
    Args:
        text: Input text
        k: Shingle size (default: 3)
        
    Returns:
        Set of k-word shingles
    """
    shingler = WordShingles(k=k)
    return shingler.generate_shingles(text)


def jaccard_similarity(text_a: str, text_b: str, k: int = 3) -> float:
    """
    Convenience function to calculate Jaccard similarity between two texts.
    
    Args:
        text_a: First text
        text_b: Second text
        k: Shingle size (default: 3)
        
    Returns:
        Jaccard similarity score (0.0 to 1.0)
    """
    shingler = WordShingles(k=k)
    shingles_a = shingler.generate_shingles(text_a)
    shingles_b = shingler.generate_shingles(text_b)
    return shingler.jaccard_similarity(shingles_a, shingles_b)
