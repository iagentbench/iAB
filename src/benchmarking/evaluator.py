"""
Evaluator module for calculating QA metrics.

This module provides functions to calculate evaluation metrics:
- Exact match (accuracy)
- F1 score
- Normalized exact match
"""

import string
from typing import List, Dict, Tuple


def normalize(text: str) -> str:
    """
    Normalize text for exact match comparison.
    
    Performs lightweight normalization including:
    - Converting to lowercase
    - Removing commas from numbers
    - Removing currency symbols ($€£¥)
    - Removing all punctuation
    - Collapsing multiple spaces
    
    Args:
        text: Input text string to normalize
        
    Returns:
        Normalized string
    """
    if not text or str(text).lower() == 'nan':
        return ""
    
    text = str(text).lower().strip()
    
    # Remove commas from numbers
    text = text.replace(',', '')
    
    # Remove currency symbols
    text = text.replace('$', '').replace('€', '').replace('£', '').replace('¥', '')
    
    # Remove all punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Collapse multiple spaces
    text = " ".join(text.split())
    
    return text


def calculate_exact_match(predicted: str, ground_truth: str, normalized: bool = True) -> bool:
    """
    Calculate exact match between predicted and ground truth answers.
    
    Args:
        predicted: Predicted answer string
        ground_truth: Ground truth answer string
        normalized: Whether to normalize text before comparison (default: True)
    
    Returns:
        True if exact match, False otherwise
    """
    if normalized:
        return normalize(predicted) == normalize(ground_truth)
    else:
        return str(predicted).strip().lower() == str(ground_truth).strip().lower()


def calculate_f1_score(predicted: str, ground_truth: str) -> float:
    """
    Calculate F1 score between predicted and ground truth answers.
    
    Uses token-level precision and recall.
    
    Args:
        predicted: Predicted answer string
        ground_truth: Ground truth answer string
    
    Returns:
        F1 score (0.0 to 1.0)
    """
    pred_tokens = set(normalize(predicted).split())
    gt_tokens = set(normalize(ground_truth).split())
    
    if len(gt_tokens) == 0:
        return 1.0 if len(pred_tokens) == 0 else 0.0
    
    if len(pred_tokens) == 0:
        return 0.0
    
    # Calculate precision and recall
    intersection = pred_tokens & gt_tokens
    precision = len(intersection) / len(pred_tokens) if len(pred_tokens) > 0 else 0.0
    recall = len(intersection) / len(gt_tokens) if len(gt_tokens) > 0 else 0.0
    
    # Calculate F1
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_accuracy(exact_matches: List[bool]) -> float:
    """
    Calculate accuracy from list of exact match results.
    
    Args:
        exact_matches: List of boolean exact match results
    
    Returns:
        Accuracy score (0.0 to 1.0)
    """
    if not exact_matches:
        return 0.0
    return sum(exact_matches) / len(exact_matches)


def calculate_metrics(
    predictions: List[str],
    ground_truths: List[str]
) -> Dict[str, float]:
    """
    Calculate all metrics for a set of predictions.
    
    Args:
        predictions: List of predicted answer strings
        ground_truths: List of ground truth answer strings
    
    Returns:
        Dictionary containing:
        - 'exact_match_accuracy': Accuracy of exact matches
        - 'exact_match_normalized_accuracy': Accuracy of normalized exact matches
        - 'f1_score': Average F1 score
        - 'exact_matches': List of exact match results (normalized)
        - 'f1_scores': List of F1 scores
    """
    if len(predictions) != len(ground_truths):
        raise ValueError(f"Predictions and ground truths must have same length. Got {len(predictions)} and {len(ground_truths)}")
    
    exact_matches = []
    exact_matches_raw = []
    f1_scores = []
    
    for pred, gt in zip(predictions, ground_truths):
        if pred:  # Only calculate if we got an answer
            exact_matches.append(calculate_exact_match(pred, gt, normalized=True))
            exact_matches_raw.append(calculate_exact_match(pred, gt, normalized=False))
            f1_scores.append(calculate_f1_score(pred, gt))
        else:
            exact_matches.append(False)
            exact_matches_raw.append(False)
            f1_scores.append(0.0)
    
    return {
        'exact_match_accuracy': calculate_accuracy(exact_matches),
        'exact_match_normalized_accuracy': calculate_accuracy(exact_matches),  # Same as above since we normalize by default
        'f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        'exact_matches': exact_matches,
        'f1_scores': f1_scores
    }


def compare_modes(
    mode1_predictions: List[str],
    mode2_predictions: List[str],
    mode3_predictions: List[str],
    ground_truths: List[str],
    mode1_name: str = "Mode 1",
    mode2_name: str = "Mode 2",
    mode3_name: str = "Mode 3"
) -> Dict:
    """
    Compare metrics across three modes.
    
    Args:
        mode1_predictions: Predictions from mode 1
        mode2_predictions: Predictions from mode 2
        mode3_predictions: Predictions from mode 3
        ground_truths: Ground truth answers
        mode1_name: Name for mode 1 (default: "Mode 1")
        mode2_name: Name for mode 2 (default: "Mode 2")
        mode3_name: Name for mode 3 (default: "Mode 3")
    
    Returns:
        Dictionary with metrics for each mode and comparative analysis
    """
    metrics1 = calculate_metrics(mode1_predictions, ground_truths)
    metrics2 = calculate_metrics(mode2_predictions, ground_truths)
    metrics3 = calculate_metrics(mode3_predictions, ground_truths)
    
    all_metrics = {
        mode1_name: metrics1,
        mode2_name: metrics2,
        mode3_name: metrics3
    }
    
    # Find best performing mode
    best_exact_match = max(
        [(mode1_name, metrics1['exact_match_accuracy']),
         (mode2_name, metrics2['exact_match_accuracy']),
         (mode3_name, metrics3['exact_match_accuracy'])],
        key=lambda x: x[1]
    )
    
    best_f1 = max(
        [(mode1_name, metrics1['f1_score']),
         (mode2_name, metrics2['f1_score']),
         (mode3_name, metrics3['f1_score'])],
        key=lambda x: x[1]
    )
    
    return {
        'metrics': all_metrics,
        'best_exact_match': {'mode': best_exact_match[0], 'score': best_exact_match[1]},
        'best_f1': {'mode': best_f1[0], 'score': best_f1[1]}
    }


# Test script: Run this file directly to test the evaluator
if __name__ == "__main__":
    print("Testing Evaluator...")
    print("=" * 60)
    
    # Test 1: Exact match
    print("\nTest 1: Exact Match")
    print("-" * 60)
    assert calculate_exact_match("Paris", "Paris") == True
    assert calculate_exact_match("Paris", "paris") == True  # Normalized
    assert calculate_exact_match("Paris", "London") == False
    print("✓ Exact match tests passed")
    
    # Test 2: F1 score
    print("\nTest 2: F1 Score")
    print("-" * 60)
    f1 = calculate_f1_score("Pierre Agostini, Ferenc Krausz, Anne L'Huillier", "Pierre Agostini")
    print(f"F1 score for partial match: {f1:.3f}")
    assert f1 > 0.0
    print("✓ F1 score tests passed")
    
    # Test 3: Calculate metrics
    print("\nTest 3: Calculate Metrics")
    print("-" * 60)
    predictions = ["Paris", "London", "Tokyo"]
    ground_truths = ["Paris", "Paris", "Tokyo"]
    metrics = calculate_metrics(predictions, ground_truths)
    print(f"Exact Match Accuracy: {metrics['exact_match_accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    assert metrics['exact_match_accuracy'] == 2/3  # 2 out of 3 correct
    print("✓ Metrics calculation tests passed")
    
    # Test 4: Compare modes
    print("\nTest 4: Compare Modes")
    print("-" * 60)
    mode1 = ["Paris", "London", "Tokyo"]
    mode2 = ["Paris", "Paris", "Tokyo"]
    mode3 = ["Paris", "Berlin", "Tokyo"]
    ground_truths = ["Paris", "Paris", "Tokyo"]
    
    comparison = compare_modes(mode1, mode2, mode3, ground_truths)
    print(f"Best Exact Match: {comparison['best_exact_match']}")
    print(f"Best F1: {comparison['best_f1']}")
    print("✓ Mode comparison tests passed")
    
    print("\n" + "=" * 60)
    print("All evaluator tests passed!")

