"""
plots.py
========
"""

import matplotlib
matplotlib.use("Agg")            # draw to a file, no extra window needed
import matplotlib.pyplot as plt

LABELS = {"weighted": "Cross-shop IDF-weighted",
          "msmp": "Title model words (MSMP representation)",
          "msmpplus": "Title + value model words (MSMP+ representation)"}
# one colour per representation (marker and line style also differ, so it is still
# readable when printed in black and white)
STYLES = {"weighted": dict(color="#c2185b", marker="o"),                   # pink
          "msmp": dict(color="#1f6fd1", marker="s", linestyle="--"),       # blue
          "msmpplus": dict(color="#1b7a3e", marker="^", linestyle=":")}    # dark green
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
        for method in ["weighted", "msmp", "msmpplus"]:   # my method first in the legend
            if method not in set(summary["method"]):
                continue
            group = summary[summary["method"] == method].sort_values("fraction")
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
            # only the diamond here, the curves are already in the F1* legend next to it
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[-1:], labels[-1:], fontsize=7, loc="lower right")
        ax.set_title(title)
        ax.set_xscale("log")      # most of the action happens at very small fractions
        ax.set_xlabel("Fraction of comparisons (log scale)")
        ax.grid(alpha=0.3)
    # one legend per row of the figure, in an empty corner, so the top half (PC, PQ)
    # and the bottom half (F1*, F1) can also be used as separate figures
    axes[0, 1].legend(fontsize=7, loc="upper right")
    axes[1, 0].legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
