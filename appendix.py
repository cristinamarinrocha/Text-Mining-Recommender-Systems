"""
appendix.py
===========
Extra figures and tables for the appendix (data inspection and diagnostics).
It uses the same modules, settings and bootstrap splits (same seed) as main.py,
so the numbers are consistent with the main results.

Usage (from inside the code folder):
    python appendix.py                  # 10 bootstraps
    python appendix.py --bootstraps 5

Output in results/appendix/:
    A1_similarity_distributions.png  pair similarity of duplicates vs
                                     non-duplicates, baseline vs paper
    A2_effect_of_a.png               PC and F1* vs fraction of comparisons,
                                     unweighted (a=0) vs weighted (a=8)
    A3_tuning_curves.png             F1 / F1* against a (incl. a=10, 12) and
                                     F1 against the probability threshold theta
    T1_tokens_filter.tex             tokens and the cross-shop filter
    T2_logit_coefficients.tex        logistic regression coefficients
    appendix_numbers.csv             the numbers behind the figures
"""

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing import load_products, product_tokens
from representation import (build_weighted_representation, build_msmp_representation,
                            document_and_shop_frequency, cross_shop_vocabulary,
                            idf_weights, multiplicities)
from lsh import MinHasher, lsh_candidate_pairs
from duplicate_detection import (PairFeatures, fit_and_tune, scored_pairs, FEATURE_NAMES,
                                 PROBABILITY_GRID, detect_code_token)
from evaluation import (bootstrap_split, true_duplicate_pairs, lsh_metrics,
                        f1_measure)
from main import N_HASHES, R_VALUES, MAX_COPIES, SEED


BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
PAPER_METHOD = dict(color=BLUE, linestyle="-", marker="o", label="Cross-shop IDF-weighted, a = 8")
BASE = dict(color=ORANGE, linestyle="--", marker="s", label="Title model words (MSMP representation)")
NOWEIGHT = dict(color=AQUA, linestyle=":", marker="^", label="Cross-shop tokens, unweighted (a = 0)")
A_EXTENDED = [0, 2, 4, 6, 8, 10, 12]
BEST_A = 8           # the value chosen by the tuning in almost every setting
R_F1, R_F1STAR = 3, 7  # settings with the best F1 and the best F1* in main.py

plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.6,
                     "lines.markersize": 4.5, "legend.frameon": False})


def splits(products, n_bootstraps):
    """Bootsrap splits as in main.py."""
    rng = np.random.default_rng(SEED)
    for _ in range(n_bootstraps):
        train_idx, test_idx = bootstrap_split(len(products), rng)
        yield [products[i] for i in train_idx], [products[i] for i in test_idx]


def lsh_jaccard(rep, i, j, m):
    """Jaccard of the expanded sets E_i, E_j"""
    shared = sum(m.get(t, 0) for t in rep.sets[i] & rep.sets[j])
    union = sum(m.get(t, 0) for t in rep.sets[i] | rep.sets[j])
    return shared / union if union else 0.0


# -------------------------------------------------------------------------
# Graph A1 - similarity of duplicates vs non-duplicates (whole data set)
# -------------------------------------------------------------------------
def figure_a1(products, out):
    n = len(products)
    true = true_duplicate_pairs(products)
    base = build_msmp_representation(products)
    paper = build_weighted_representation(products, BEST_A, MAX_COPIES)
    m = multiplicities(paper.weights, MAX_COPIES)
    base_m = {t: 1 for t in base.weights}
    sims = {"base": ([], []), "paper": ([], [])}    # (duplicates, non-duplicates)
    for i, j in combinations(range(n), 2):
        k = 0 if (i, j) in true else 1
        sims["base"][k].append(lsh_jaccard(base, i, j, base_m))
        sims["paper"][k].append(lsh_jaccard(paper, i, j, m))

    b = N_HASHES // R_F1
    t = (1 / b) ** (1 / R_F1)   # s-curve threshold of the r=3 setting
    # bins of width 0.1: Jaccard values of small sets are few distinct fractions
    bins = np.linspace(0, 1, 11)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    rows = []
    for ax, key, style, title in [(axes[0], "base", BASE, "Title model words (MSMP representation)"),
                                  (axes[1], "paper", PAPER_METHOD, "Cross-shop IDF-weighted, a = 8")]:
        dup, non = (np.array(x) for x in sims[key])
        for values, ls, lab in [(dup, "-", "Duplicate pairs"), (non, "--", "Non-duplicate pairs")]:
            share = np.histogram(values, bins=bins)[0] / len(values)
            share = np.where(share > 0, share, np.nan)            # empty bins not drawn
            ax.stairs(share, bins, color=style["color"], linestyle=ls, linewidth=1.6,
                      label=f"{lab} (n = {len(values):,})")
        ax.axvline(t, color="0.35", linewidth=1, linestyle="-.")
        ax.text(t + 0.015, 0.03, f"t ≈ {t:.2f}\n(r = {R_F1}, b = {b})", fontsize=7,
                color="0.25", va="bottom", transform=ax.get_xaxis_transform())
        above_dup, above_non = (dup >= t).mean(), (non >= t).mean()
        ax.text(0.98, 0.97, f"share with similarity ≥ t:\nduplicates {above_dup:.1%}\n"
                f"non-duplicates {above_non:.2%}", transform=ax.transAxes, ha="right",
                va="top", fontsize=7, color="0.15",
                bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.3"))
        ax.set_yscale("log")
        ax.set_xlim(0, 1)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Similarity estimated by min-hashing (Jaccard of the LSH sets)")
        ax.grid(alpha=0.25)
        rows.append({"figure": "A1", "representation": key, "share_dup_above_t": above_dup,
                     "share_nondup_above_t": above_non,
                     "median_dup": np.median(dup), "median_nondup": np.median(non)})
    axes[0].set_ylabel("Share of the pairs in the class (log scale)")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color="0.2", linestyle="-", label="Duplicate pairs (399)"),
                        Line2D([], [], color="0.2", linestyle="--",
                               label="Non-duplicate pairs (1,317,477)"),
                        Line2D([], [], color="0.35", linestyle="-.", linewidth=1,
                               label="LSH threshold t of the setting r = 3")],
               loc="lower center", ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    # (no suptitle: the LaTeX caption describes the figure)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "A1_similarity_distributions.png", dpi=200)
    plt.close(fig)
    return rows


# -----------------------------------------------------------------------
# Graphs A2 + A3 + T2 - per bootstrap
# -----------------------------------------------------------------------

def bootstrap_exps(products, n_bootstraps):
    hasher = MinHasher(N_HASHES, seed=SEED)
    lsh_rows, tune_rows, theta_rows, coef_rows = [], [], [], []
    for boot, (train_p, test_p) in enumerate(splits(products, n_bootstraps)):
        print(f"  bootstrap {boot + 1}/{n_bootstraps}")
        sets = {}
        for name, sub in (("train", train_p), ("test", test_p)):
            sets[name] = {"shops": [p.shop for p in sub], "true": true_duplicate_pairs(sub),
                          "features": PairFeatures(sub), "n": len(sub), "products": sub}

        def signatures(name, a):
            sub = sets[name]["products"]
            rep = (build_msmp_representation(sub) if a is None
                   else build_weighted_representation(sub, a, MAX_COPIES))
            return hasher.signature_matrix(rep.elements)

        # Graph A2: LSH quality for the baseline, a = 0 and a = 8 (test set, all r)
        for label, a in (("base", None), ("a0", 0), ("a8", BEST_A)):
            sig = signatures("test", a)
            for r in R_VALUES:
                c = lsh_candidate_pairs(sig, N_HASHES // r, r)
                row = {"bootstrap": boot, "series": label, "r": r}
                row.update(lsh_metrics(c, sets["test"]["true"], sets["test"]["n"]))
                lsh_rows.append(row)

        # Graph A3: F1 (r=3) and F1* (r=7) for every a, incl. a = 10, 12
        for a in A_EXTENDED:
            sig_tr, sig_te = signatures("train", a), signatures("test", a)
            c_tr = lsh_candidate_pairs(sig_tr, N_HASHES // R_F1, R_F1)
            c_te = lsh_candidate_pairs(sig_te, N_HASHES // R_F1, R_F1)
            tr, te = sets["train"], sets["test"]
            model, theta, f1_train = fit_and_tune(c_tr, tr["features"], tr["shops"], tr["true"])
            accepted = scored_pairs(c_te, te["features"], te["shops"], model)
            f1_test = f1_measure({p for p, q in accepted.items() if q >= theta}, te["true"])["f1"]
            c7 = lsh_candidate_pairs(sig_te, N_HASHES // R_F1STAR, R_F1STAR)
            f1s = lsh_metrics(c7, te["true"], te["n"])["F1_star"]
            tune_rows.append({"bootstrap": boot, "a": a, "f1_train": f1_train,
                              "f1_test": f1_test, "F1_star_r7": f1s})

            # Tunning parameter graph --> theta curve and coefficients at the best setting (a=8, r=3)
            if a == BEST_A and model is not None:
                acc_tr = scored_pairs(c_tr, tr["features"], tr["shops"], model)
                for th in PROBABILITY_GRID:
                    theta_rows.append({
                        "bootstrap": boot, "theta": th, "chosen": th == theta,
                        "f1_train": f1_measure({p for p, q in acc_tr.items() if q >= th}, tr["true"])["f1"],
                        "f1_test": f1_measure({p for p, q in accepted.items() if q >= th}, te["true"])["f1"]})
                clf = model["clf"]          # scikit-learn LogisticRegression
                coef_rows.append(dict(zip(["intercept"] + FEATURE_NAMES,
                                          [clf.intercept_[0], *clf.coef_[0]]),
                                      bootstrap=boot))
    return (pd.DataFrame(lsh_rows), pd.DataFrame(tune_rows),
            pd.DataFrame(theta_rows), pd.DataFrame(coef_rows))


def band(ax, x, mean, sd, style):
    ax.plot(x, mean, **style)
    ax.fill_between(x, mean - sd, mean + sd, color=style["color"], alpha=0.15, linewidth=0)


def plot_a2(lsh, out):
    g = lsh.groupby(["series", "r"]).agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for ax, metric, title in [(axes[0], "PC", "Pair completeness"),
                              (axes[1], "F1_star", "F1*-measure")]:
        for key, style in (("base", BASE), ("a0", NOWEIGHT), ("a8", PAPER_METHOD)):
            d = g[g.series == key].sort_values(("fraction", "mean"))
            band(ax, d[("fraction", "mean")].values, d[(metric, "mean")].values,
                 d[(metric, "std")].fillna(0).values, style)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Fraction of comparisons (log scale)")
        ax.grid(alpha=0.25)
    axes[1].legend(fontsize=7, loc="upper right")
    # (no suptitle: the LaTeX caption describes the figure)
    fig.tight_layout()
    fig.savefig(out / "A2_effect_of_a.png", dpi=200)
    plt.close(fig)


def plot_a3(tune, theta, out):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
    g = tune.groupby("a").agg(["mean", "std"])
    a = g.index.values
    band(axes[0], a, g[("f1_test", "mean")].values, g[("f1_test", "std")].values,
         dict(PAPER_METHOD, label="Test F1"))
    axes[0].plot(a, g[("f1_train", "mean")].values, color=AQUA, linestyle=":", marker="^",
                 label="Train F1 (used for tuning)")
    axes[0].set_title(f"F1 against a  (r = {R_F1}, b = {N_HASHES // R_F1})", fontsize=9)
    band(axes[1], a, g[("F1_star_r7", "mean")].values, g[("F1_star_r7", "std")].values,
         dict(PAPER_METHOD, label="Test F1*"))
    axes[1].set_title(f"F1* against a  (r = {R_F1STAR}, b = {N_HASHES // R_F1STAR})", fontsize=9)
    for ax in axes[:2]:
        ax.axvspan(9, 12.5, color="0.92", zorder=0)
        ax.text(10.75, ax.get_ylim()[0], "outside\nmain grid", ha="center", va="bottom",
                fontsize=7, color="0.35")
        ax.set_xticks(A_EXTENDED)
        ax.set_xlabel("Exponent a of the weights w(t) = idf(t)^a")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, loc="upper left")
    t = theta.groupby("theta").agg(["mean", "std"])
    band(axes[2], t.index.values, t[("f1_test", "mean")].values,
         t[("f1_test", "std")].values, dict(PAPER_METHOD, label="Test F1", markersize=3))
    axes[2].plot(t.index.values, t[("f1_train", "mean")].values, color=AQUA, linestyle=":",
                 marker="^", markersize=3, label="Train F1 (used for tuning)")
    chosen = theta[theta.chosen].theta
    axes[2].axvspan(chosen.min(), chosen.max(), color="0.9", zorder=0,
                    label=f"range of chosen θ ({chosen.min():.2f}–{chosen.max():.2f})")
    axes[2].set_title(f"F1 against θ  (a = {BEST_A}, r = {R_F1})", fontsize=9)
    axes[2].set_xlabel("Probability threshold θ")
    axes[2].grid(alpha=0.25)
    axes[2].legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("F1 / F1*")
    # (no suptitle: the LaTeX caption describes the figure)
    fig.tight_layout()
    fig.savefig(out / "A3_tuning_curves.png", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
# T1 - tokens and the cross-shop filter (whole data set, no labels)
# -------------------------------------------------------------------------
def table_t1(products, out):
    raw = [product_tokens(p) for p in products]
    df, sf = document_and_shop_frequency(raw, [p.shop for p in products])
    V = cross_shop_vocabulary(df, sf)
    once = {t for t in df if df[t] == 1}
    one_shop = {t for t in df if df[t] >= 2 and sf[t] == 1}
    w = idf_weights(V, df, len(products), BEST_A)
    m = multiplicities(w, MAX_COPIES)
    zero = {t for t in V if m[t] == 0}
    per_product_raw = np.mean([len(s) for s in raw])
    per_product_v = np.mean([len(s & V) for s in raw])
    per_product_m = np.mean([len({t for t in s & V if m[t] > 0}) for s in raw])
    common = sorted(V, key=lambda t: -df[t])[:6]
    rare = [t for t in sorted(V, key=lambda t: df[t]) if detect_code_token(t)][:6]
    rows = [("Distinct tokens (title + values)", f"{len(df):,}"),
            ("\\quad removed: in only one product ($\\mathrm{df}=1$)", f"{len(once):,}"),
            ("\\quad removed: in only one shop ($\\mathrm{df}\\ge2$, $\\mathrm{sf}=1$)", f"{len(one_shop):,}"),
            ("Kept vocabulary $|V|$", f"{len(V):,}"),
            (f"\\quad of which 0 copies at $a={BEST_A}$ ($m(t)=0$)", f"{len(zero):,}"),
            ("Tokens per product: raw / after filter / with $\\ge1$ copy",
             f"{per_product_raw:.1f} / {per_product_v:.1f} / {per_product_m:.1f}"),
            ("Most common kept tokens (lowest weight)", ", ".join(f"\\texttt{{{t}}}" for t in common)),
            ("Rarest kept code-like tokens (highest weight)", ", ".join(f"\\texttt{{{t}}}" for t in rare))]
    tex = ["\\begin{table}", "\\centering",
           "\\caption{Tokens and the cross-shop filter on the full data set "
           f"(1,624 products; $a={BEST_A}$, $M={MAX_COPIES}$).}}", "\\label{tab:tokens}",
           "\\begin{tabular}{lr}", "\\hline"]
    tex += [f"{k} & {v} \\\\" for k, v in rows[:6]] + ["\\hline"]
    tex += [f"\\multicolumn{{2}}{{l}}{{{k}: {v}}} \\\\" for k, v in rows[6:]]
    tex += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (out / "T1_tokens_filter.tex").write_text("\n".join(tex), encoding="utf-8")
    return [{"figure": "T1", "item": k, "value": v} for k, v in rows]


# -------------------------------------------------------------------------
# T2 - logistic regression coefficients (a=8, r=3)
# -------------------------------------------------------------------------
def table_t2(coefs, out):
    names = {"intercept": "Intercept", "title_jaccard": "Title similarity (Jaccard)", "code_match": "Shared code",
             "code_conflict": "Code conflict", "unit_conflicts": "Unit conflicts"}
    tex = ["\\begin{table}", "\\centering",
           f"\\caption{{Logistic regression coefficients (standardised features; $a={BEST_A}$, "
           f"$r={R_F1}$), mean (sd) over {len(coefs)} bootstraps. A coefficient is the change "
           "in log-odds of a duplicate for a one-standard-deviation increase of the feature.}",
           "\\label{tab:coefficients}", "\\begin{tabular}{lrr}", "\\hline",
           "Feature & Coefficient & Odds ratio \\\\", "\\hline"]
    for key, label in names.items():
        mean, sd = coefs[key].mean(), coefs[key].std()
        odds = "" if key == "intercept" else f"{np.exp(mean):.2f}"
        tex.append(f"{label} & {mean:+.2f} ({sd:.2f}) & {odds} \\\\")
    tex += ["\\hline", "\\end{tabular}", "\\end{table}"]
    (out / "T2_logit_coefficients.tex").write_text("\n".join(tex), encoding="utf-8")


if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=here.parent / "TVs-all-merged.json", type=Path)
    parser.add_argument("--bootstraps", default=10, type=int)
    parser.add_argument("--out", default=here / "results" / "appendix", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    products = load_products(args.data)
    print("A1: similarity distributions (all pairs) ...")
    numbers = figure_a1(products, args.out)
    print("T1: tokens and filter ...")
    numbers += table_t1(products, args.out)
    print("A2, A3, T2: bootstraps ...")
    lsh, tune, theta, coefs = bootstrap_exps(products, args.bootstraps)
    plot_a2(lsh, args.out)
    plot_a3(tune, theta, args.out)
    table_t2(coefs, args.out)

    pd.DataFrame(numbers).to_csv(args.out / "appendix_numbers.csv", index=False)
    lsh.to_csv(args.out / "A2_lsh_per_bootstrap.csv", index=False)
    tune.to_csv(args.out / "A3_a_per_bootstrap.csv", index=False)
    theta.to_csv(args.out / "A3_theta_per_bootstrap.csv", index=False)
    coefs.to_csv(args.out / "T2_coefficients_per_bootstrap.csv", index=False)
    print("\nF1 / F1* against a (mean over bootstraps):")
    print(tune.groupby("a")[["f1_train", "f1_test", "F1_star_r7"]].mean().round(3))
    print(f"\nSaved figures and tables in {args.out}")
