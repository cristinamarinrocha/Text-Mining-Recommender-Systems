# Duplicate detection with cross-shop IDF-weighted tokens and LSH

Individual assignment for Text Mining & Recommender Systems

Cristina Marín Rocha (681737).

## The project

The data has 1,624 TV descriptions from four web shops (Amazon, Newegg, Best Buy, TheNerds). 
Comparing all pairs (about 1.3 million) does not scale, LSH is used (own implementation) to select candidate pairs and only compare those.

The main idea is the representation that LSH runs on:

- each product is the set of tokens from its title and attribute values (keys are ignored, so nothing depends on the TV attributes)
- a token is only kept if it appears in at least two products and at least two shops
- each token gets a weight `w(t) = idf(t)^a`, so rare tokens like model codes count a lot and common ones like `1080p` or `hdtv` almost not
- the weights are put into the sets by repeating every token `m(t)` times (proportional to its weight), so normal min-hashing estimates the weighted Jaccard similarity

The candidate pairs are then classified with a logistic regression on 4 features from the titles (title similarity, shared code, conflicting code, conflicting units), followed by a clustering step (MSM paper style) so the predicted duplicates are transitive.

As baselines I use the MSMP representation and the MSMP+ representation in the same pipeline with the same classifier, so the only difference is the representation.

## Files

The step numbers in the comments of the code follow the method:

- `preprocessing.py`: loading the data, normalisation and tokenisation (steps 0-2)
- `representation.py`: cross-shop filter, idf weights, weighted Jaccard and token copies (steps 3-6a), plus the MSMP baseline
- `lsh.py`: min-hashing and banding (steps 6b-6c)
- `duplicate_detection.py`: step 7, both classifiers. Part 1 is the first one I used (a threshold on `J_w` for pairs from different shops, `--classifier threshold`), part 2 is the default (pair features, logistic regression with scikit-learn, clustering and the probability threshold)
- `evaluation.py`: bootstrap split, PQ, PC, F1*, fraction of comparisons and F1 (step 8)
- `plots.py`: the main figure
- `appendix.py`: figures and tables for the appendix of the paper
- `main.py`: runs everything

## Set-up

- 10 bootstraps: the drawn products are the training set, the out-of-bag products the test set
- signature length `n = 600`, rows per band `r` in {1,...,10,12,15,20}, `b = n // r`
- for every `(b, r)`, the exponent `a` (0, 2, 4, 6 or 8), the logistic regression and the threshold `θ` are chosen on the training set (best F1) and then used unchanged on the test set
- `modelID` is only used to know which pairs are duplicates (for training and evaluation), never as input

## Running it

Needs Python 3.9 or newer and packages: `numpy`, `pandas`, `matplotlib`, `scikit-learn`:

**Main.py :**

```bash
pip install numpy pandas matplotlib scikit-learn
cd code
python main.py                          # full run, 10 bootstraps
python main.py --bootstraps 1           # quick test
python main.py --all-pairs              # also classify all pairs without LSH (reference)
python main.py --classifier threshold   # old classifier, results go to results_threshold/
```

Other options: `--data path/to/TVs-all-merged.json` (default: one folder above `code/`) and `--out folder` (default: `results/`). Options can be combined.

The full run takes a while because every `(b, r)` setting is tuned separately. Output in `results/`:

- `results_per_bootstrap.csv`: all measures per bootstrap and setting
- `results_summary.csv`: averages over the bootstraps
- `results_plots.png`: PC, PQ, F1* and F1 against the fraction of comparisons (band = mean ± 1 sd)
- `results_all_pairs.csv`: F1 without LSH (only with `--all-pairs`)

**Appendix.py :**

Appendix (same bootstrap splits as `main.py`):

```bash
python appendix.py                  # results in results/appendix/
python appendix.py --bootstraps 5
```

The output are the following figures: `A1` (similarity of duplicates vs non-duplicates), `A2` (with vs without weights), `
A3` (tuning of `a` and `θ`), the two LaTeX tables `T1` (tokens and filter) and `T2` (coefficients), and csv files with the numbers behind them.
