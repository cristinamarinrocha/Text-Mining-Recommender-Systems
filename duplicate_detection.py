"""
duplicate_detection.py
======================
STEP 7 - Duplicate detection on the LSH candidate pairs only.

Two classifiers, chosen with  python main.py --classifier logit|threshold :

PART 1 - Threshold rule
PART 2 (default) - logistic regression followed by clustering
    7a. Pair features  x(i, j)          (generic, no product knowledge)
    7b. Logistic regression             P(duplicate | x)
    7c. Clustering
    7d. Probability threshold theta     tuned on F1 on the training sample
"""

import re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from preprocessing import normalise, tokenise
from representation import weighted_jaccard
from evaluation import f1_measure


# PART 1 - threshold
# ---------------------------------------------------------------------------
THETA_GRID = np.round(np.arange(0.05, 1.00, 0.025), 3)

def candidate_similarities(candidates, rep, shops, exclude_same_shop=True):
    """Exact weighted Jaccard similarity for every candidate pair that is
    allowed to be a duplicate (pairs from the same shop get similarity 0)."""
    sims = {}
    for i, j in candidates:
        if exclude_same_shop and shops[i] == shops[j]:
            sims[(i, j)] = 0.0
        else:
            sims[(i, j)] = weighted_jaccard(rep, i, j)
    return sims


def predict(similarities, theta):
    return {pair for pair, s in similarities.items() if s >= theta}


def tune_threshold(similarities, true_pairs, theta_grid=THETA_GRID):
    """Keep best theta."""
    best_theta, best_f1 = theta_grid[0], -1.0
    for theta in theta_grid:
        f1 = f1_measure(predict(similarities, theta), true_pairs)["f1"]
        if f1 > best_f1:
            best_theta, best_f1 = theta, f1
    return best_theta, best_f1


# PART 2 (default) - logistic regression + clustering
# ---------------------------------------------------------------------------
PROBABILITY_GRID = np.round(np.arange(0.05, 1.00, 0.025), 3)
FEATURE_NAMES = ["title_jaccard", "code_match", "code_conflict",
                 "unit_conflicts"] # covariates of logit

# A "unit token" is a number followed by letters only, e.g. 46inch, 120hz, 1080p.
UNIT_TOKEN = re.compile(r"^(\d+(?:\.\d+)?)([a-z]+)$") # will exclude from "code"


def detect_code_token(token):
    """
    Code-like token:
    - at least 5 characters
    - contains letters and digits
    - is not a unit token (number + unit).
    """
    return (len(token) >= 5 and any(ch.isdigit() for ch in token)
            and any(ch.isalpha() for ch in token)
            and UNIT_TOKEN.match(token) is None)


# -------------------------------------------------------------------------
# STEP 7a - Pair features
# -------------------------------------------------------------------------
class PairFeatures:
    """
    Pre-computes, for one subset of products (train or test), everything the
    features need. It caches the feature vector of every pair it has seen
    for efficiency in calculations (the same pair appears for many (b, r) settings).
    """

    def __init__(self, products):
        self.title_tokens = [tokenise(normalise(p.title)) for p in products]
        self.codes = [{t for t in s if detect_code_token(t)} for s in self.title_tokens]
        self.units = []
        for s in self.title_tokens:
            units = {}
            for t in s:
                m = UNIT_TOKEN.match(t) # exclude unit tokens from code like ones
                if m:
                    units.setdefault(m.group(2), set()).add(m.group(1))
            self.units.append(units)
        self._cache = {}

    def _pair(self, i, j):
        # (A content-similarity feature, the idf-weighted Jaccard of all tokens,
        #  was removed: it repeats what LSH already selected on, coefficient ~0.)
        # 1. Title Similarity
        # plain Jaccard of the title tokens
        ti, tj = self.title_tokens[i], self.title_tokens[j]
        title_jaccard = len(ti & tj) / len(ti | tj) if ti | tj else 0.0
        # 2./3. Shared code and Code conflict
        # code-like tokens: shared one (strong evidence for a duplicate)
        #  or both have codes but none in common (evidence against)
        ci, cj = self.codes[i], self.codes[j]
        code_match = 1.0 if ci & cj else 0.0
        code_conflict = 1.0 if (ci and cj and not ci & cj) else 0.0
        # 4. Unit conflicts
        # number of units that both titles mention with different numbers
        # so eveidence against duplicate (e.g. 46inch vs 55inch)
        ui, uj = self.units[i], self.units[j]
        unit_conflicts = float(sum(1 for u in ui.keys() & uj.keys()
                                   if not ui[u] & uj[u]))
        return (title_jaccard, code_match, code_conflict, unit_conflicts)

    def cov_matrix(self, pairs):
        """Covariate matrix containing all features for logit.
         One row per pair (in the order of `pairs`)."""
        rows = []
        for pair in pairs:
            x = self._cache.get(pair)
            if x is None:
                x = self._pair(*pair)
                self._cache[pair] = x
            rows.append(x)
        return np.array(rows, dtype=float).reshape(-1, len(FEATURE_NAMES))


# -------------------------------------------------------------------------
# STEP 7b - Logistic regression
# -------------------------------------------------------------------------
def fit_logistic(X, y):
    if y.sum() < 2 or (len(y) - y.sum()) < 2:
        return None
    scaler = StandardScaler().fit(X)
    # C in the apckage is like a lamda in ridge penalty
    # C=100 --> Lambda=0.01
    clf = LogisticRegression(C=100, solver="lbfgs", max_iter=1000)
    clf.fit(scaler.transform(X), y)
    return {"scaler": scaler, "clf": clf}

def predict_proba(model, X):
    return model["clf"].predict_proba(model["scaler"].transform(X))[:, 1]

# -------------------------------------------------------------------------
# STEP 7c - Clustering
# -------------------------------------------------------------------------
def cluster_pairs(pairs, probs, shops, candidates, min_prob=PROBABILITY_GRID[0]):
    """
    Pairwise predictions are not transitive by themselves, so we put
    the products in clusters and every pair inside a cluster is a duplicate.
    Distance between two products = 1 - probability.
    Go through the pairs from the highest to the lowest probability (= single linkage) and
    merge the clusters of i and j, except when (distance = infinity in MSM):
      - both clusters have a product of the same shop  (same shop heuristic)
      - some pair between the two clusters is not an LSH candidate pair
        (avoid prediction of pairs we never compared)
    Returns
    {pair: probability at which the pair ended in the same cluster}.
    """
    candidates = set(candidates)
    cluster_of = {}   # product -> id of its cluster
    members = {}      # cluster id -> products in it
    cluster_shops = {}  # cluster id -> shops in it
    accepted = {}
    for k in np.argsort(-probs):
        if probs[k] < min_prob:
            break
        i, j = pairs[k]
        # products not merged yet are a cluster on their own
        for x in (i, j):
            if x not in cluster_of:
                cluster_of[x], members[x], cluster_shops[x] = x, [x], {shops[x]}
        ci, cj = cluster_of[i], cluster_of[j]
        if ci == cj:  # already in the same cluster
            continue
        if cluster_shops[ci] & cluster_shops[cj]:  # same shop twice -> not allowed
            continue
        new_pairs = [(min(x, y), max(x, y)) for x in members[ci] for y in members[cj]]
        if any(pair not in candidates for pair in new_pairs):  # not compared -> not allowed
            continue
        # merge cj into ci, all new pairs are duplicates from this probability on
        for pair in new_pairs:
            accepted[pair] = probs[k]
        for x in members[cj]:
            cluster_of[x] = ci
        members[ci] += members.pop(cj)
        cluster_shops[ci] |= cluster_shops.pop(cj)
    return accepted

# -------------------------------------------------------------------------
# STEP 7d - Train and apply
# -------------------------------------------------------------------------
def scored_pairs(candidates, features, shops, model):
    """
    This function combines steps 7b and 7c.
    Pairs that end in the same cluster, with their probability."""
    pairs = [(i, j) for i, j in candidates if shops[i] != shops[j]]
    if model is None or not pairs:
        return {}
    probs = predict_proba(model, features.cov_matrix(pairs)) # 7b
    return cluster_pairs(pairs, probs, shops, candidates) # 7c


def fit_and_tune(candidates, features, shops, true_pairs, grid=PROBABILITY_GRID):
    """TRAIN: fit the logistic regression on the training candidates and choose
    the probability threshold with the best F1.
    Returns (model, theta, f1)."""
    pairs = [(i, j) for i, j in candidates if shops[i] != shops[j]]
    if not pairs:
        return None, grid[0], 0.0
    X = features.cov_matrix(pairs)
    y = np.array([pair in true_pairs for pair in pairs], dtype=float)
    model = fit_logistic(X, y)
    if model is None:
        return None, grid[0], 0.0
    accepted = cluster_pairs(pairs, predict_proba(model, X), shops, candidates)
    # tunning
    best_theta, best_f1 = grid[0], -1.0
    for theta in grid:
        predicted = {pair for pair, p in accepted.items() if p >= theta}
        f1 = f1_measure(predicted, true_pairs)["f1"]
        if f1 > best_f1:
            best_theta, best_f1 = theta, f1
    return model, best_theta, best_f1 # will use these settings on test sample


def classify(candidates, features, shops, model, theta):
    """TEST: predicted duplicate pairs."""
    accepted = scored_pairs(candidates, features, shops, model)
    return {pair for pair, p in accepted.items() if p >= theta}
