"""
representation.py
=================
Build the product representation:
STEP 3 - Cross-shop filtering   keep tokens with df(t) >= 2 and sf(t) >= 2
STEP 4 - Rarity weights         w(t) = idf(t)^a
STEP 5 - Weighted Jaccard       J_w(p_i, p_j)
STEP 6a - Copies                each token t becomes m(t) copies "t#1", ..., "t#m(t)"

It also contains the BASELINE representation (MSMP: title model words)..
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from preprocessing import product_tokens


@dataclass
class Representation:
    """Everything the rest of the pipeline needs about a set of products.
    Called as rep along the script"""
    sets: list          # sets[i]      = filtered token set  ~T_i (step 3)
    weights: dict       # weights[t]   = w(t)  (1.0 for every token in the baseline) (step 4)
    elements: list      # elements[i]  = list of strings fed to minhash (with copies) (step 6a)
    total_weight: list  # total_weight[i] = sum of w(t) over ~T_i (speeds up Step 5)


# -------------------------------------------------------------------------
# STEP 3 - Cross-shop filtering
# -------------------------------------------------------------------------
def document_and_shop_frequency(token_sets, shops):
    """
    df[t] = in how many products token t appears
    sf[t] = in how many DIFFERENT shops token t appears
    """
    df = Counter()
    shops_of_token = defaultdict(set)
    for tokens, shop in zip(token_sets, shops):
        for t in tokens:
            df[t] += 1
            shops_of_token[t].add(shop)
    sf = {t: len(s) for t, s in shops_of_token.items()}
    return df, sf


def cross_shop_vocabulary(df, sf, min_df=2, min_sf=2):
    """V = tokens that can link two products of two different shops."""
    # toten that occurs in only one shop reflects taht shop's wording
    # rather than a product charactereistic
    return {t for t in df if df[t] >= min_df and sf[t] >= min_sf}


# ------------------------------------------------------------------------
# STEP 4 - Rarity weights
# ------------------------------------------------------------------------
def idf_weights(vocabulary, df, n_products, a):
    """w(t) = idf(t)^a  with  idf(t) = log(N / df(t))"""
    # loop over all tokens in the set and find their weight
    # note here hyperparameter a enters (the exponent determining how strongly
    # rare tokens are favoured in the weighted representation)
    return {t: math.log(n_products / df[t]) ** a for t in vocabulary}


# -------------------------------------------------------------------------
# STEP 5 - Weighted Jaccard similarity
# -------------------------------------------------------------------------
def weighted_jaccard(rep, i, j):
    """
    J_w = (sum of weights of SHARED tokens) / (sum of weights of ALL tokens of i or j).
    Use that weight(union) = weight(i) + weight(j) - weight(intersection),
    so only need to loop over the (small) intersection.
    """
    shared = sum(rep.weights[t] for t in rep.sets[i] & rep.sets[j]) # numerator: tokens both products have
    union = rep.total_weight[i] + rep.total_weight[j] - shared # denominator
    return shared / union if union > 0 else 0.0


# -------------------------------------------------------------------------
# STEP 6a - Integer copies
# -------------------------------------------------------------------------
def multiplicities(weights, max_copies):
    """
    m(t) = round(c * w(t))
    c = max_copies / max_t w(t)
    Tokens whose weight is below 1/(2*max_copies) of the heaviest weight round
    to 0 copies.
    """
    if not weights:
        return {}
    # w_max --> weight of rarets token (htos ein only 2 products)
    # w_min --> most common tokwn weight
    w_max, w_min = max(weights.values()), min(weights.values())
    if math.isclose(w_max, w_min): # nothing to copy if no tokens at all.
        return {t: 1 for t in weights}
    c = max_copies / w_max
    # return m(t)
    return {t: round(c * w) for t, w in weights.items()}


def expand_with_copies(tokens, m):
    """E_i: build set where each token is replaced with m dsitinct copies
    Plain Jaccard on these expanded sets = weighted Jaccard with weights m(t)."""
    return [f"{t}#{l}" for t in tokens for l in range(1, m[t] + 1)]


# --------------------------------------------------------------------------
# Combine steps 1-6a
# --------------------------------------------------------------------------
def build_weighted_representation(products, a, max_copies=30):
    """ Pipeline steps 1-6a"""
    raw_sets = [product_tokens(p) for p in products]                     # Steps 1-2
    df, sf = document_and_shop_frequency(raw_sets, [p.shop for p in products])
    vocabulary = cross_shop_vocabulary(df, sf)                           # Step 3
    sets = [tokens & vocabulary for tokens in raw_sets] # filtered token set  ~T_i
    weights = idf_weights(vocabulary, df, len(products), a)              # Step 4
    m = multiplicities(weights, max_copies)                              # Step 6a
    elements = [expand_with_copies(s, m) for s in sets]
    total = [sum(weights[t] for t in s) for s in sets]
    return Representation(sets, weights, elements, total)


# -------------------------------------------------------------------------
# BASELINE - MSMP title model words
# -------------------------------------------------------------------------
# A model word contains both digits and non-digits, e.g. "120hz", "ln46c630".
MODEL_WORD = re.compile(r"[a-z0-9]*(?:[0-9]+[^0-9, ]+|[^0-9, ]+[0-9]+)[a-z0-9]*")

def build_msmp_representation(products):
    """Binary set of model words from the title, no filtering."""
    sets = [set(MODEL_WORD.findall(p.title.lower())) for p in products]
    vocabulary = set().union(*sets)
    weights = {t: 1.0 for t in vocabulary}
    elements = [list(s) for s in sets]
    total = [float(len(s)) for s in sets]
    return Representation(sets, weights, elements, total)
