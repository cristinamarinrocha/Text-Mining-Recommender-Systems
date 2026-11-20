"""
plots.py
========
"""

import matplotlib
matplotlib.use("Agg")            # draw to a file, no extra window needed
import matplotlib.pyplot as plt

LABELS = {"weighted": "Cross-shop IDF-weighted",
          "msmp": "Title model words (MSMP representation)"}
STYLES = {"weighted": dict(color="black", marker="o"),
          "msmp": dict(color="grey", marker="s", linestyle="--")}
LABELLED_R = [1, 4, 7, 12, 20]   # easy read in plots --> r values whose points get an "r=..." label


def plot_results(summary, path, per_bootstrap=None, all_pairs=None):
    """
    - per_bootstrap: the results of every bootstrap; if given, the
    spread over the bootstraps is drawn as a shaded band.
    - all_pairs: the no-LSH results per bootstrap (optional); their mean
    F1 (+/- one standard deviation) is drawn at fraction 1 in the F1 panel.
    It is the same for both methods (same classifier, no LSH)."""
    if per_bootstrap is not None:
        spread = per_bootstrap.groupby(["method", "b", "r"]).std(numeric_only=True)
    metrics = [("PC", "Pair completeness"), ("PQ", "Pair quality"),
               ("F1_star", "F1*-measure"), ("f1", "F1-measure")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for ax, (column, title) in zip(axes.ravel(), metrics):
        for method, group in summary.groupby("method"):
            group = group.sort_values("fraction")
            ax.plot(group["fraction"], group[column], label=LABELS[method],
                    markersize=3, **STYLES[method])
            if per_bootstrap is not None:
                sd = spread.loc[method].loc[list(zip(group["b"], group["r"])), column].values
                ax.fill_between(group["fraction"], group[column] - sd, group[column] + sd,
                                color=STYLES[method]["color"], alpha=0.15, linewidth=0)
            for _, point in group[group["r"].isin(LABELLED_R)].iterrows():
                ax.annotate(f"r={int(point['r'])}", (point["fraction"], point[column]),
                            textcoords="offset points", xytext=(3, 3), fontsize=6,
                            color=STYLES[method]["color"])
        if column == "f1" and all_pairs is not None:
            f1 = all_pairs.drop_duplicates("bootstrap")["f1"]   # identical for both methods
            ax.errorbar([1.0], [f1.mean()], yerr=[f1.std()], fmt="D", color="black",
                        markerfacecolor="white", markersize=5, capsize=3,
                        label="All pairs, no LSH")
            ax.legend(fontsize=7, loc="lower right")
        ax.set_title(title)
        ax.set_xscale("log")      # most of the action happens at very small fractions
        ax.set_xlabel("Fraction of comparisons (log scale)")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
