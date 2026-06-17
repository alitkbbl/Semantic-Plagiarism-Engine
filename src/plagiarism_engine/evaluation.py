"""
Evaluation utilities for plagiarism detection methods.

Provides:
- Metrics computation (precision, recall, F1, accuracy)
- Speed benchmarking
- Comparison of different methods
"""

import time
from typing import List, Dict, Tuple, Callable, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate classification metrics from predictions."""
    
    @staticmethod
    def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> Dict[str, int]:
        """
        Compute confusion matrix counts.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
        
        Returns:
            Dictionary with TP, FP, TN, FN counts
        """
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")
        
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
        tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
        
        return {
            'true_positives': tp,
            'false_positives': fp,
            'true_negatives': tn,
            'false_negatives': fn
        }
    
    @staticmethod
    def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """
        Compute precision, recall, F1-score, and accuracy.
        
        Args:
            y_true: Ground truth labels (0 or 1)
            y_pred: Predicted labels (0 or 1)
        
        Returns:
            Dictionary with precision, recall, f1_score, accuracy
        """
        cm = MetricsCalculator.compute_confusion_matrix(y_true, y_pred)
        
        tp = cm['true_positives']
        fp = cm['false_positives']
        tn = cm['true_negatives']
        fn = cm['false_negatives']
        
        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # F1-score: 2 * (precision * recall) / (precision + recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Accuracy: (TP + TN) / (TP + TN + FP + FN)
        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            **cm
        }
    
    @staticmethod
    def apply_threshold(similarities: List[float], threshold: float) -> List[int]:
        """
        Convert similarity scores to binary predictions.
        
        Args:
            similarities: List of similarity scores (0.0 to 1.0)
            threshold: Threshold for classification
        
        Returns:
            List of binary predictions (0 or 1)
        """
        return [1 if sim >= threshold else 0 for sim in similarities]


class BenchmarkRunner:
    """Run and compare different plagiarism detection methods."""
    
    def __init__(self, pairs: List[Dict]):
        """
        Initialize benchmark runner.
        
        Args:
            pairs: List of pair dictionaries with 'text_a', 'text_b', 'label'
        """
        self.pairs = pairs
        self.results = []
    
    def run_method(self,
                   method_name: str,
                   similarity_fn: Callable[[str, str], float],
                   threshold: float = 0.5,
                   **method_params) -> Dict:
        """
        Run a similarity method on all pairs and compute metrics.
        
        Args:
            method_name: Name of the method (e.g., 'jaccard', 'minhash', 'simhash')
            similarity_fn: Function that takes two texts and returns similarity score
            threshold: Threshold for classification
            **method_params: Additional method parameters to log
        
        Returns:
            Dictionary with metrics and timing information
        """
        logger.info(f"Running method: {method_name}")
        
        y_true = [pair['label'] for pair in self.pairs]
        similarities = []
        
        start_time = time.time()
        
        for i, pair in enumerate(self.pairs):
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(self.pairs)} pairs")
            
            try:
                sim = similarity_fn(pair['text_a'], pair['text_b'])
                similarities.append(sim)
            except Exception as e:
                logger.warning(f"Error computing similarity for pair {i}: {e}")
                similarities.append(0.0)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Convert similarities to predictions
        y_pred = MetricsCalculator.apply_threshold(similarities, threshold)
        
        # Compute metrics
        metrics = MetricsCalculator.compute_metrics(y_true, y_pred)
        
        result = {
            'method': method_name,
            'threshold': threshold,
            'execution_time_seconds': execution_time,
            'pairs_per_second': len(self.pairs) / execution_time if execution_time > 0 else 0.0,
            **metrics,
            **method_params
        }
        
        self.results.append(result)
        
        logger.info(f"  Completed in {execution_time:.2f}s")
        logger.info(f"  Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}")
        
        return result
    
    def get_results(self) -> List[Dict]:
        """
        Get all benchmark results.
        
        Returns:
            List of result dictionaries
        """
        return self.results
    
    def compare_methods(self) -> Dict:
        """
        Compare all methods and find the best performing one.
        
        Returns:
            Dictionary with comparison summary
        """
        if not self.results:
            return {}
        
        # Find best method by F1-score
        best_f1 = max(self.results, key=lambda x: x['f1_score'])
        
        # Find fastest method
        fastest = min(self.results, key=lambda x: x['execution_time_seconds'])
        
        return {
            'best_f1_method': best_f1['method'],
            'best_f1_score': best_f1['f1_score'],
            'fastest_method': fastest['method'],
            'fastest_time_seconds': fastest['execution_time_seconds'],
            'all_results': self.results
        }


def evaluate_method(pairs: List[Dict],
                    similarity_fn: Callable[[str, str], float],
                    threshold: float = 0.5) -> Dict[str, float]:
    """
    Convenience function to evaluate a single method.
    
    Args:
        pairs: List of pair dictionaries with 'text_a', 'text_b', 'label'
        similarity_fn: Function that computes similarity between two texts
        threshold: Classification threshold
    
    Returns:
        Dictionary with metrics
    """
    y_true = [pair['label'] for pair in pairs]
    similarities = [similarity_fn(pair['text_a'], pair['text_b']) for pair in pairs]
    y_pred = MetricsCalculator.apply_threshold(similarities, threshold)
    
    return MetricsCalculator.compute_metrics(y_true, y_pred)


def benchmark_speed(similarity_fn: Callable[[str, str], float],
                   text_pairs: List[Tuple[str, str]],
                   num_iterations: int = 1) -> Dict[str, float]:
    """
    Benchmark speed of a similarity function.
    
    Args:
        similarity_fn: Function to benchmark
        text_pairs: List of (text_a, text_b) tuples
        num_iterations: Number of times to repeat (for averaging)
    
    Returns:
        Dictionary with timing statistics
    """
    total_time = 0.0
    
    for iteration in range(num_iterations):
        start_time = time.time()
        
        for text_a, text_b in text_pairs:
            _ = similarity_fn(text_a, text_b)
        
        end_time = time.time()
        total_time += (end_time - start_time)
    
    avg_time = total_time / num_iterations
    pairs_per_second = len(text_pairs) / avg_time if avg_time > 0 else 0.0
    
    return {
        'total_time_seconds': total_time,
        'average_time_seconds': avg_time,
        'pairs_per_second': pairs_per_second,
        'num_pairs': len(text_pairs),
        'num_iterations': num_iterations
    }
