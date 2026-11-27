"""
main.py
=======

Pipeline per bootstrap and per LSH setting (b, r):
    1. TRAIN set  -> build representation, minhash, LSH, tune the parameters
                     (exponent a and the step-7 classifier) on F1
    2. TEST set   -> same pipeline with the tuned parameters
                  -> PQ, PC, F1*, fraction of comparisons (LSH step)
                  -> F1 (final duplicate detection)
My method "weighted" and the baselines "msmp" (title model words) and
"msmpplus" (MSMP+: title + value model words) are evaluated, with the
same step-7 classifier.
"""

import argparse
import time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd

from preprocessing import load_products
from representation import (build_weighted_representation, build_msmp_representation,
                            build_msmpplus_representation)
from lsh import MinHasher, lsh_candidate_pairs, s_curve_threshold
from duplicate_detection import (candidate_similarities, predict, tune_threshold, THETA_GRID,
                                 PairFeatures, fit_and_tune, classify)
from evaluation import (bootstrap_split, true_duplicate_pairs, lsh_metrics,
                        f1_measure)
from plots import plot_results

# -------------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------------
N_HASHES = 600                     # signature length n
R_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]   # rows per band; b = n // r
A_VALUES = [0, 2, 4, 6, 8]        # candidate exponents for w(t) = idf(t)^a
MAX_COPIES = 30                   # copies of the heaviest token (step 6a)
SEED = 2026

def build_representation(method, products, a):
    if method == "weighted":
        return build_weighted_representation(products, a, MAX_COPIES)
    if method == "msmpplus":
        return build_msmpplus_representation(products)  # second baseline (MSMP+)
    return build_msmp_representation(products)


class SubsetPipeline:
    """
    Holds one subset of products (train or test) and gives, for every value of
    a, its representation and signature matrix.
    This is computed only once and reused for all (b, r) settings.
    """

    def __init__(self, products, method, hasher):
        self.products = products
        self.method = method
        self.hasher = hasher
        self.shops = [p.shop for p in products]
        self.true_pairs = true_duplicate_pairs(products)
        self._cache = {} # dictionary that stores values we have already computed
        self._features = None

    def representation_and_signatures(self, a):
        if a not in self._cache: # in first round rep and sig aren't  in self._cache yet so we build them
            rep = build_representation(self.method, self.products, a) # representation
            sig = self.hasher.signature_matrix(rep.elements) # signature
            self._cache[a] = (rep, sig)         # store and reuse in later rounds (b, r)
        return self._cache[a]

    @property
    def features(self):
        """Pair features of the step 7 classifier (built once)."""
        if self._features is None:
            self._features = PairFeatures(self.products)
        return self._features

# -------------------------------------------------------------------------
# Step 7 - choose classifier (old threshold rule or new logistic regression)
# -------------------------------------------------------------------------
def tune_step7(classifier, cands, rep, subset):
    """TRAIN: returns (model, theta, f1).
    - model is None for the threshold rule."""
    if classifier == "threshold":
        sims = candidate_similarities(cands, rep, subset.shops)
        theta, f1 = tune_threshold(sims, subset.true_pairs)
        return None, theta, f1
    return fit_and_tune(cands, subset.features, subset.shops, subset.true_pairs) # logit + clustering classifier


def apply_step7(classifier, cands, rep, subset, model, theta):
    """TEST: returns the set of predicted duplicate pairs."""
    if classifier == "threshold":
        return predict(candidate_similarities(cands, rep, subset.shops), theta)
    return classify(cands, subset.features, subset.shops, model, theta) # logit classifier


def classify_all_pairs(boot, method, a_values, train, test, classifier):
    """
    Reference WITHOUT LSH: every pair is a candidate (fraction of comparisons = 1).
    a and the classifier are tuned on the train set same as with LSH, then
    applied to the test set.
    Shows the best F1 the classifier can reach (how much F1 is lost by using LSH).
    """
    def all_pairs_of(subset, rep):
        pairs = list(combinations(range(len(subset.products)), 2))
        if classifier == "threshold":
            # pairs below the smallest theta can never be predicted as duplicates,
            # drop them early for faster result
            sims = candidate_similarities(pairs, rep, subset.shops)
            pairs = [pair for pair, s in sims.items() if s >= THETA_GRID[0]]
        return pairs

    # with the logit classifier a only changes the LSH candidates, not the features,
    if classifier != "threshold":
        a_values = a_values[:1]  # without LSH one value of a is enough
    best = {"f1": -1.0}
    for a in a_values:
        rep, _ = train.representation_and_signatures(a)
        model, theta, f1 = tune_step7(classifier, all_pairs_of(train, rep), rep, train)
        if f1 > best["f1"]:
            best = {"a": a, "model": model, "theta": theta, "f1": f1} #tune

    rep, _ = test.representation_and_signatures(best["a"]) # fit
    predicted = apply_step7(classifier, all_pairs_of(test, rep), rep, test,
                            best["model"], best["theta"])
    row = {"bootstrap": boot, "method": method, "classifier": classifier,
           "a": best["a"], "theta": best["theta"]}
    row.update(f1_measure(predicted, test.true_pairs))
    print(f"  {method:8s} ALL PAIRS (no LSH)  F1={row['f1']:.3f}")
    return row


def main_run(data_path, n_bootstraps, out_dir, classifier="logit", all_pairs=False):
    products = load_products(data_path)
    print(f"Loaded {len(products)} products.")
    rng = np.random.default_rng(SEED)
    hasher = MinHasher(N_HASHES, seed=SEED)
    rows = []
    all_pairs_rows = []   # only filled when all_pairs=True

    # BOOTSTRAP
    for boot in range(n_bootstraps):
        # draw bootstrap sample
        train_idx, test_idx = bootstrap_split(len(products), rng)
        train_products = [products[i] for i in train_idx]
        test_products = [products[i] for i in test_idx]
        print(f"\nBootstrap {boot + 1}/{n_bootstraps}: "
              f"{len(train_products)} train, {len(test_products)} test products")

        # Loop over my weighted method and the two baselines
        for method in ["weighted", "msmp", "msmpplus"]:
            a_values = A_VALUES if method == "weighted" else [None]
            #split train and test
            train = SubsetPipeline(train_products, method, hasher)
            test = SubsetPipeline(test_products, method, hasher)

            for r in R_VALUES:
                b = N_HASHES // r

                # 1. TRAIN: choose a and the classifier (theta) with the best F1 (CV)
                best = {"f1": -1.0} #initialize
                for a in a_values:
                    rep, sig = train.representation_and_signatures(a) # build representation on train set
                    cands = lsh_candidate_pairs(sig, b, r)
                    model, theta, f1 = tune_step7(classifier, cands, rep, train) #logit classifier
                    if f1 > best["f1"]:
                        best = {"a": a, "model": model, "theta": theta, "f1": f1}

                # 2. TEST: apply the tuned settings
                start = time.perf_counter() # for the record
                rep, sig = test.representation_and_signatures(best["a"]) # build representation on test set
                cands = lsh_candidate_pairs(sig, b, r)
                lsh_seconds = time.perf_counter() - start

                predicted = apply_step7(classifier, cands, rep, test,
                                        best["model"], best["theta"])

                row = {"bootstrap": boot, "method": method,
                       "classifier": classifier, "b": b, "r": r,
                       "s_curve_t": s_curve_threshold(b, r),
                       "a": best["a"], "theta": best["theta"],
                       "lsh_seconds": lsh_seconds}
                row.update(lsh_metrics(cands, test.true_pairs, len(test_products))) # main results
                row.update(f1_measure(predicted, test.true_pairs))
                rows.append(row)
                print(f"  {method:8s} b={b:3d} r={r:2d}  frac={row['fraction']:.4f}  "
                      f"PC={row['PC']:.3f}  PQ={row['PQ']:.3f}  "
                      f"F1*={row['F1_star']:.3f}  F1={row['f1']:.3f}  (a={best['a']})")

            # Optional all_pairs: no LSH, classify all pairs of the test set
            # (with the logit the result does not depend on the representation, so
            # no need to run it again for msmpplus)
            if all_pairs and not (method == "msmpplus" and classifier == "logit"):
                all_pairs_rows.append(
                    classify_all_pairs(boot, method, a_values, train, test, classifier))

    # 3. Save and average over bootstraps
    out_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "results_per_bootstrap.csv", index=False)

    summary = (results.groupby(["method", "b", "r"])
               [["fraction", "PQ", "PC", "F1_star", "precision", "recall", "f1",
                 "lsh_seconds"]].mean().reset_index().sort_values(["method", "fraction"]))
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    ref = pd.DataFrame(all_pairs_rows) if all_pairs else None   # no-LSH reference
    plot_results(summary, out_dir / "results_plots.png", results, ref)

    if all_pairs:
        ref.to_csv(out_dir / "results_all_pairs.csv", index=False)
        print("\nAll pairs without LSH (average over bootstraps):")
        print(ref.groupby("method")[["precision", "recall", "f1"]].mean().round(3))
    print(f"\nSaved results and plots in {out_dir}")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=here.parent / "TVs-all-merged.json", type=Path)
    parser.add_argument("--bootstraps", default=10, type=int)
    parser.add_argument("--out", default=None, type=Path,
                        help="output folder (default: results/, or "
                             "results_threshold/ for the old classifier)")
    parser.add_argument("--classifier", default="logit", choices=["logit", "threshold"],
                        help="step 7: logistic regression (new) or J_w threshold (old)")
    parser.add_argument("--all-pairs", action="store_true",
                        help="also classify all pairs without LSH (reference F1)")
    args = parser.parse_args()
    out = args.out or here / ("results" if args.classifier == "logit"
                              else "results_threshold")
    main_run(args.data, args.bootstraps, out, args.classifier, args.all_pairs)
