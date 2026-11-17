"""
evaluation.py
=============
STEP 8 - Evaluation

* Bootstrapping
* LSH (scalability) metrics:
      pair quality      PQ  = duplicates found / comparisons made
      pair completeness PC  = duplicates found / total duplicates
      F1*               = harmonic mean of PQ and PC
      fraction of comparisons = comparisons made / all possible comparisons
* F1 of the predicted duplicate pairs.

"""

from collections import defaultdict
from itertools import combinations

import numpy as np


def bootstrap_split(n_products, rng):
    """Returns (train_indices, test_indices) of one bootstrap.
    Bootstrap gives build in train and test sets (OOB evaluations)"""
    drawn = rng.integers(0, n_products, size=n_products)   # Bootstrap is with replacement
    train = np.unique(drawn)
    test = np.setdiff1d(np.arange(n_products), train)
    return train, test


def true_duplicate_pairs(products):
    """All pairs (i, j), i < j, of products with the same modelID (ground truth)."""
    by_model = defaultdict(list)
    for i, p in enumerate(products):
        by_model[p.model_id].append(i)
    pairs = set()
    for members in by_model.values():
        pairs.update(combinations(members, 2))
    return pairs


def harmonic_mean(x, y):
    # For F1*
    return 2 * x * y / (x + y) if x + y > 0 else 0.0


def lsh_metrics(candidates, true_pairs, n_products):
    """Build PQ, PC, F1* and fraction of comparisons of the LSH step."""
    total_comparisons = n_products * (n_products - 1) // 2
    n_comparisons = len(candidates)
    found = len(candidates & true_pairs)
    pq = found / n_comparisons if n_comparisons else 0.0
    pc = found / len(true_pairs) if true_pairs else 0.0
    return {"comparisons": n_comparisons,
            "fraction": n_comparisons / total_comparisons,
            "PQ": pq, "PC": pc, "F1_star": harmonic_mean(pq, pc)}


def f1_measure(predicted_pairs, true_pairs):
    """Pair-level precision, recall and F1
    - Using TP/FP/FN """
    tp = len(predicted_pairs & true_pairs)
    fp = len(predicted_pairs) - tp
    fn = len(true_pairs) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": harmonic_mean(precision, recall)}
